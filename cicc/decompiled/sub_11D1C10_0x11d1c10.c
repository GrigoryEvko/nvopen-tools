// Function: sub_11D1C10
// Address: 0x11d1c10
//
__int64 __fastcall sub_11D1C10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _BYTE *v6; // rax
  unsigned int v7; // r13d
  _QWORD *v8; // r12
  _BYTE *v9; // rbx
  _QWORD *v10; // rdi
  __int64 v12; // [rsp+0h] [rbp-B0h] BYREF
  __int64 v13; // [rsp+8h] [rbp-A8h]
  _QWORD *v14; // [rsp+10h] [rbp-A0h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-98h]
  _BYTE v16[32]; // [rsp+90h] [rbp-20h] BYREF

  v6 = &v14;
  v12 = 0;
  v13 = 1;
  do
  {
    *(_QWORD *)v6 = -4096;
    v6 += 32;
  }
  while ( v6 != v16 );
  v7 = sub_11D0CC0(a1, a2 * 8, a3, a4, (__int64)&v12, a6);
  if ( (v13 & 1) != 0 )
  {
    v9 = v16;
    v8 = &v14;
  }
  else
  {
    v8 = v14;
    a2 = 4LL * v15;
    if ( !v15 )
      goto LABEL_14;
    v9 = &v14[a2];
    if ( &v14[a2] == v14 )
      goto LABEL_14;
  }
  do
  {
    if ( *v8 != -8192 && *v8 != -4096 )
    {
      v10 = (_QWORD *)v8[1];
      if ( v10 != v8 + 3 )
        _libc_free(v10, a2 * 8);
    }
    v8 += 4;
  }
  while ( v8 != (_QWORD *)v9 );
  if ( (v13 & 1) == 0 )
  {
    v8 = v14;
    a2 = 4LL * v15;
LABEL_14:
    sub_C7D6A0((__int64)v8, a2 * 8, 8);
  }
  return v7;
}
