// Function: sub_C367B0
// Address: 0xc367b0
//
__int64 __fastcall sub_C367B0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  unsigned int v4; // r14d
  __int64 v6; // r15
  unsigned int v7; // eax
  unsigned int v8; // r10d
  unsigned int v9; // ecx
  unsigned int v10; // eax
  __int64 v11; // r8
  int v12; // r10d
  __int64 v13; // rcx
  unsigned int v14; // r9d
  int v15; // r10d
  int v17; // eax
  unsigned int v18; // [rsp+4h] [rbp-3Ch]
  unsigned int v19; // [rsp+4h] [rbp-3Ch]
  int v20; // [rsp+8h] [rbp-38h]
  unsigned int v21; // [rsp+8h] [rbp-38h]
  unsigned int v22; // [rsp+8h] [rbp-38h]
  unsigned int v23; // [rsp+Ch] [rbp-34h]
  int v24; // [rsp+Ch] [rbp-34h]

  v4 = a3;
  *(_BYTE *)(a1 + 20) = *(_BYTE *)(a1 + 20) & 0xF8 | 2;
  v20 = sub_C45E30(a2, (unsigned int)a3, a3);
  v6 = sub_C33900(a1);
  v7 = sub_C337D0(a1);
  v8 = v20 + 1;
  v9 = *(_DWORD *)(*(_QWORD *)a1 + 8LL);
  if ( v20 + 1 < v9 )
  {
    *(_DWORD *)(a1 + 16) = v9 - 1;
    sub_C49830(v6, v7, a2, v8, 0);
    v15 = 0;
  }
  else
  {
    v18 = v7;
    *(_DWORD *)(a1 + 16) = v20;
    v21 = v9;
    v23 = v8 - v9;
    v10 = sub_C45DF0(a2, v4);
    v11 = v23;
    v12 = 0;
    v13 = v21;
    v14 = v18;
    if ( v23 > v10 )
    {
      if ( v23 == v10 + 1 )
      {
        v12 = 2;
      }
      else if ( v23 > v4 << 6
             || (v19 = v21, v22 = v14, v17 = sub_C45D90(a2, v23 - 1), v11 = v23, v14 = v22, v12 = 3, v13 = v19, !v17) )
      {
        v12 = 1;
      }
    }
    v24 = v12;
    sub_C49830(v6, v14, a2, v13, v11);
    v15 = v24;
  }
  return sub_C36450(a1, a4, v15);
}
