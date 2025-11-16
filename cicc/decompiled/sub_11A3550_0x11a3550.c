// Function: sub_11A3550
// Address: 0x11a3550
//
__int64 __fastcall sub_11A3550(__int64 a1, __int64 a2, unsigned int a3, unsigned int a4, __int64 a5, unsigned int a6)
{
  __int64 v6; // rbx
  __int64 *v7; // rbx
  __int64 v8; // r12
  unsigned __int8 *v9; // r14
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v16[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v6 = *(_QWORD *)(a2 - 8);
  else
    v6 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v7 = (__int64 *)(32LL * a3 + v6);
  v8 = sub_11A3690(a1, *v7, a4, a5, a6, a2);
  if ( !v8 )
    return 0;
  v9 = (unsigned __int8 *)*v7;
  if ( *(_BYTE *)*v7 <= 0x1Cu || (sub_F54ED0((unsigned __int8 *)*v7), (v9 = (unsigned __int8 *)*v7) != 0) )
  {
    v10 = v7[1];
    *(_QWORD *)v7[2] = v10;
    if ( v10 )
      *(_QWORD *)(v10 + 16) = v7[2];
  }
  *v7 = v8;
  v11 = *(_QWORD *)(v8 + 16);
  v7[1] = v11;
  if ( v11 )
    *(_QWORD *)(v11 + 16) = v7 + 1;
  v7[2] = v8 + 16;
  *(_QWORD *)(v8 + 16) = v7;
  if ( *v9 > 0x1Cu )
  {
    v12 = *(_QWORD *)(a1 + 40);
    v16[0] = (__int64)v9;
    v13 = v12 + 2096;
    sub_11A2F60(v13, v16);
    v14 = *((_QWORD *)v9 + 2);
    if ( v14 )
    {
      if ( !*(_QWORD *)(v14 + 8) )
      {
        v16[0] = *(_QWORD *)(v14 + 24);
        sub_11A2F60(v13, v16);
      }
    }
  }
  return 1;
}
