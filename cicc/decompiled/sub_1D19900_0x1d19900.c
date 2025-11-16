// Function: sub_1D19900
// Address: 0x1d19900
//
bool __fastcall sub_1D19900(__int64 a1, __int64 a2, __int64 a3, int a4, __int64 a5, __int64 a6)
{
  __int64 *v7; // rax
  __int64 v8; // rdx
  __int64 v10; // rcx
  __int64 v12; // rax
  char v14; // di
  __int64 v15; // rax
  unsigned int v16; // eax
  int v17; // r8d
  int v18; // [rsp-BCh] [rbp-BCh]
  int v19; // [rsp-BCh] [rbp-BCh]
  __int64 v20; // [rsp-B0h] [rbp-B0h] BYREF
  char v21; // [rsp-A8h] [rbp-A8h] BYREF
  __int64 v22; // [rsp-A0h] [rbp-A0h]
  _BYTE v23[48]; // [rsp-98h] [rbp-98h] BYREF
  _BYTE v24[104]; // [rsp-68h] [rbp-68h] BYREF

  if ( (*(_BYTE *)(a2 + 26) & 8) != 0 )
    return 0;
  if ( (*(_BYTE *)(a3 + 26) & 8) == 0 && (*(_WORD *)(a2 + 26) & 0x380) == 0 && (*(_WORD *)(a3 + 26) & 0x380) == 0 )
  {
    v7 = *(__int64 **)(a3 + 32);
    v8 = *(_QWORD *)(a2 + 32);
    v10 = *v7;
    if ( *(_QWORD *)v8 == *v7 && *(_DWORD *)(v8 + 8) == *((_DWORD *)v7 + 2) )
    {
      v12 = *(_QWORD *)(a2 + 40);
      v14 = *(_BYTE *)v12;
      v15 = *(_QWORD *)(v12 + 8);
      v21 = v14;
      v22 = v15;
      if ( v14 )
      {
        v16 = sub_1D13440(v14);
      }
      else
      {
        v19 = a5;
        v16 = sub_1F58D40(&v21, a2, v8, v10, a5, a6);
        v17 = v19;
      }
      if ( v16 >> 3 == a4 )
      {
        v18 = v17;
        sub_2043720(v23, a3, a1);
        sub_2043720(v24, a2, a1);
        v20 = 0;
        if ( (unsigned __int8)sub_2043540(v23, v24, a1, &v20) )
          return a4 * v18 == v20;
      }
    }
  }
  return 0;
}
