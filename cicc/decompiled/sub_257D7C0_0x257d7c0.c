// Function: sub_257D7C0
// Address: 0x257d7c0
//
_BOOL8 __fastcall sub_257D7C0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r12
  __int64 v4; // r13
  unsigned __int64 *v5; // r10
  unsigned __int64 *v6; // r13
  unsigned __int64 *v7; // r15
  int v8; // r12d
  __int64 (__fastcall *v9)(__int64); // rdx
  __int64 v10; // rax
  unsigned __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rax
  _BOOL4 v14; // r12d
  int v16; // eax
  int v17; // r12d
  unsigned __int64 v18; // r13
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned __int64 v21; // r12
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  unsigned __int64 *v24; // rax
  char v25; // [rsp+27h] [rbp-71h] BYREF
  unsigned __int64 *v26; // [rsp+28h] [rbp-70h] BYREF
  __int64 v27; // [rsp+30h] [rbp-68h]
  _BYTE v28[96]; // [rsp+38h] [rbp-60h] BYREF

  v3 = (_QWORD *)(a1 + 72);
  v26 = (unsigned __int64 *)v28;
  v27 = 0x300000000LL;
  v25 = 0;
  if ( (unsigned __int8)sub_2526B50(a2, (const __m128i *)(a1 + 72), a1, (__int64)&v26, 3u, &v25, 1u) )
  {
    v4 = (unsigned int)v27;
  }
  else
  {
    v18 = sub_250D070(v3);
    v21 = sub_2509740(v3);
    v22 = (unsigned int)v27;
    v23 = (unsigned int)v27 + 1LL;
    if ( v23 > HIDWORD(v27) )
    {
      sub_C8D5F0((__int64)&v26, v28, v23, 0x10u, v19, v20);
      v22 = (unsigned int)v27;
    }
    v24 = &v26[2 * v22];
    *v24 = v18;
    v24[1] = v21;
    v4 = (unsigned int)(v27 + 1);
    LODWORD(v27) = v27 + 1;
  }
  v5 = v26;
  v6 = &v26[2 * v4];
  if ( v6 == v26 )
  {
    v8 = 1023;
  }
  else
  {
    v7 = v26;
    v8 = 1023;
    do
    {
      v11 = sub_250D2C0(*v7, 0);
      v13 = sub_257C550(a2, v11, v12, a1, 0, 0, 1);
      if ( !v13
        || a1 == v13
        || ((v9 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v13 + 48LL), v9 != sub_2534FF0)
          ? (v10 = v9(v13))
          : (v10 = v13 + 88),
            (v8 &= *(_DWORD *)(v10 + 12)) == 0) )
      {
        v5 = v26;
        v14 = 0;
        *(_DWORD *)(a1 + 100) = *(_DWORD *)(a1 + 96);
        goto LABEL_12;
      }
      v7 += 2;
    }
    while ( v6 != v7 );
    v5 = v26;
  }
  v16 = *(_DWORD *)(a1 + 100);
  v17 = *(_DWORD *)(a1 + 96) | v16 & v8;
  *(_DWORD *)(a1 + 100) = v17;
  v14 = v16 == v17;
LABEL_12:
  if ( v5 != (unsigned __int64 *)v28 )
    _libc_free((unsigned __int64)v5);
  return v14;
}
