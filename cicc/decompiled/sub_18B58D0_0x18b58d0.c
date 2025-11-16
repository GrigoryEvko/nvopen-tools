// Function: sub_18B58D0
// Address: 0x18b58d0
//
__int64 __fastcall sub_18B58D0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  __int64 v3; // rsi
  unsigned int v4; // eax
  char *v5; // rbx
  unsigned int v6; // r13d
  char *v7; // r12
  char *v8; // rdi
  __int64 v10; // [rsp+8h] [rbp-228h] BYREF
  __m128i v11[2]; // [rsp+10h] [rbp-220h] BYREF
  _QWORD v12[11]; // [rsp+30h] [rbp-200h] BYREF
  char *v13; // [rsp+88h] [rbp-1A8h]
  unsigned int v14; // [rsp+90h] [rbp-1A0h]
  char v15; // [rsp+98h] [rbp-198h] BYREF

  if ( a1 + 24 == (*(_QWORD *)(a1 + 24) & 0xFFFFFFFFFFFFFFF8LL) )
    return 0;
  v1 = *(_QWORD *)(a1 + 32);
  if ( !v1 )
    BUG();
  if ( v1 + 16 == (*(_QWORD *)(v1 + 16) & 0xFFFFFFFFFFFFFFF8LL) )
  {
    return 0;
  }
  else
  {
    v2 = *(_QWORD *)(v1 + 24);
    v10 = 0;
    if ( v2 )
      v2 -= 24;
    sub_15C9090((__int64)v11, &v10);
    sub_15CA330((__int64)v12, (__int64)"wholeprogramdevirt", (__int64)byte_3F871B3, 0, v11, v2);
    v3 = v10;
    if ( v10 )
      sub_161E7C0((__int64)&v10, v10);
    v4 = sub_15C8000((__int64)v12, v3);
    v5 = v13;
    v6 = v4;
    v12[0] = &unk_49ECF68;
    v7 = &v13[88 * v14];
    if ( v13 != v7 )
    {
      do
      {
        v7 -= 88;
        v8 = (char *)*((_QWORD *)v7 + 4);
        if ( v8 != v7 + 48 )
          j_j___libc_free_0(v8, *((_QWORD *)v7 + 6) + 1LL);
        if ( *(char **)v7 != v7 + 16 )
          j_j___libc_free_0(*(_QWORD *)v7, *((_QWORD *)v7 + 2) + 1LL);
      }
      while ( v5 != v7 );
      v7 = v13;
    }
    if ( v7 != &v15 )
      _libc_free((unsigned __int64)v7);
  }
  return v6;
}
