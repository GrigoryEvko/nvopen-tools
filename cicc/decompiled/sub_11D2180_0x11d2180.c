// Function: sub_11D2180
// Address: 0x11d2180
//
__int64 __fastcall sub_11D2180(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _BYTE *v7; // rax
  int v9; // ebx
  __int64 *v10; // r8
  __int64 *v11; // r12
  __int64 v12; // rdi
  int v13; // eax
  unsigned int v14; // ebx
  _QWORD *v15; // r12
  _BYTE *v16; // r13
  _QWORD *v17; // rdi
  __int64 *i; // [rsp+10h] [rbp-D0h]
  __int64 *v21; // [rsp+18h] [rbp-C8h]
  __int64 v22; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v23; // [rsp+28h] [rbp-B8h]
  _QWORD *v24; // [rsp+30h] [rbp-B0h] BYREF
  unsigned int v25; // [rsp+38h] [rbp-A8h]
  _BYTE v26[48]; // [rsp+B0h] [rbp-30h] BYREF

  v7 = &v24;
  v22 = 0;
  v23 = 1;
  do
  {
    *(_QWORD *)v7 = -4096;
    v7 += 32;
  }
  while ( v7 != v26 );
  v9 = 0;
  v10 = &v22;
  v11 = *(__int64 **)(a1 + 8);
  for ( i = *(__int64 **)(a1 + 16); i != v11; v9 |= v13 )
  {
    v12 = *v11;
    v21 = v10;
    ++v11;
    LOBYTE(v13) = sub_11D1D20(v12, a2, a3, a4, (__int64)v10, a6);
    v10 = v21;
  }
  v14 = sub_11D0CC0(a1, a2, a3, a4, (__int64)v10, a6) | v9;
  if ( (v23 & 1) != 0 )
  {
    v16 = v26;
    v15 = &v24;
  }
  else
  {
    v15 = v24;
    a2 = 32LL * v25;
    if ( !v25 )
      goto LABEL_16;
    v16 = &v24[(unsigned __int64)a2 / 8];
    if ( &v24[(unsigned __int64)a2 / 8] == v24 )
      goto LABEL_16;
  }
  do
  {
    if ( *v15 != -4096 && *v15 != -8192 )
    {
      v17 = (_QWORD *)v15[1];
      if ( v17 != v15 + 3 )
        _libc_free(v17, a2);
    }
    v15 += 4;
  }
  while ( v15 != (_QWORD *)v16 );
  if ( (v23 & 1) == 0 )
  {
    v15 = v24;
    a2 = 32LL * v25;
LABEL_16:
    sub_C7D6A0((__int64)v15, a2, 8);
  }
  return v14;
}
