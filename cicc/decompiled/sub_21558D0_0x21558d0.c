// Function: sub_21558D0
// Address: 0x21558d0
//
__int64 __fastcall sub_21558D0(__int64 a1, __int64 a2)
{
  __int64 v4; // r13
  _QWORD *v5; // r13
  __int64 *v6; // r13
  __int64 *v7; // rdi
  __int64 v8; // rdi
  _BYTE *v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r12
  _QWORD *v13; // rdi
  _BYTE *v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rax
  _QWORD *v18; // rdi
  void *s1; // [rsp+0h] [rbp-50h] BYREF
  size_t n; // [rsp+8h] [rbp-48h]
  _QWORD v21[8]; // [rsp+10h] [rbp-40h] BYREF

  v4 = *(_QWORD *)(a1 + 904);
  if ( !v4 )
  {
    v15 = *(_BYTE **)a2;
    v16 = *(_QWORD *)(a2 + 8);
    s1 = v21;
    sub_214ADD0((__int64 *)&s1, v15, (__int64)&v15[v16]);
    v17 = sub_22077B0(96);
    v4 = v17;
    if ( v17 )
      sub_21556A0(v17, (unsigned int *)&s1);
    v18 = s1;
    *(_QWORD *)(a1 + 904) = v4;
    if ( v18 != v21 )
    {
      j_j___libc_free_0(v18, v21[0] + 1LL);
      v4 = *(_QWORD *)(a1 + 904);
    }
  }
  s1 = v21;
  sub_214ADD0((__int64 *)&s1, *(_BYTE **)(v4 + 64), *(_QWORD *)(v4 + 64) + *(_QWORD *)(v4 + 72));
  v5 = s1;
  if ( n == *(_QWORD *)(a2 + 8) && (!n || !memcmp(s1, *(const void **)a2, n)) )
  {
    if ( v5 != v21 )
      j_j___libc_free_0(v5, v21[0] + 1LL);
    return *(_QWORD *)(a1 + 904);
  }
  else
  {
    if ( v5 != v21 )
      j_j___libc_free_0(v5, v21[0] + 1LL);
    v6 = *(__int64 **)(a1 + 904);
    if ( v6 )
    {
      v7 = (__int64 *)v6[8];
      if ( v7 != v6 + 10 )
        j_j___libc_free_0(v7, v6[10] + 1);
      v8 = v6[7];
      if ( v8 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v8 + 8LL))(v8);
      sub_214B5A0(v6[2]);
      j_j___libc_free_0(v6, 96);
    }
    v9 = *(_BYTE **)a2;
    v10 = *(_QWORD *)(a2 + 8);
    s1 = v21;
    sub_214ADD0((__int64 *)&s1, v9, (__int64)&v9[v10]);
    v11 = sub_22077B0(96);
    v12 = v11;
    if ( v11 )
      sub_21556A0(v11, (unsigned int *)&s1);
    v13 = s1;
    *(_QWORD *)(a1 + 904) = v12;
    if ( v13 != v21 )
    {
      j_j___libc_free_0(v13, v21[0] + 1LL);
      return *(_QWORD *)(a1 + 904);
    }
  }
  return v12;
}
