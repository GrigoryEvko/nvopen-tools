// Function: sub_2527850
// Address: 0x2527850
//
unsigned __int64 __fastcall sub_2527850(__int64 a1, __m128i *a2, __int64 a3, _BYTE *a4, unsigned __int8 a5)
{
  int v8; // edx
  __int64 v9; // r10
  __int64 v10; // rdi
  __int64 v11; // r8
  __int64 v12; // rdx
  int v13; // r15d
  unsigned int i; // eax
  __int64 v15; // rsi
  unsigned int v16; // eax
  unsigned __int64 v17; // rax
  __int64 v18; // rdx
  bool v19; // zf
  __int64 v20; // rax
  unsigned __int64 v21; // rbx
  __int64 v22; // rdx
  _BYTE *v23; // r12
  void (__fastcall *v24)(_BYTE *, _BYTE *, __int64); // rax
  _BYTE *v26; // [rsp+0h] [rbp-C0h]
  __int64 v28; // [rsp+28h] [rbp-98h] BYREF
  unsigned __int64 v29; // [rsp+30h] [rbp-90h]
  __int64 v30; // [rsp+38h] [rbp-88h]
  __int64 v31; // [rsp+40h] [rbp-80h]
  __int64 v32; // [rsp+48h] [rbp-78h]
  _BYTE *v33; // [rsp+50h] [rbp-70h] BYREF
  __int64 v34; // [rsp+58h] [rbp-68h]
  _BYTE v35[96]; // [rsp+60h] [rbp-60h] BYREF

  v8 = *(_DWORD *)(a1 + 56);
  v9 = *(_QWORD *)(a1 + 40);
  if ( !v8 )
    goto LABEL_8;
  v10 = a2->m128i_i64[0];
  v11 = a2->m128i_i64[1];
  v12 = (unsigned int)(v8 - 1);
  v13 = 1;
  for ( i = v12
          & (((unsigned int)v11 >> 9)
           ^ ((unsigned int)v11 >> 4)
           ^ (16 * (((unsigned int)a2->m128i_i64[0] >> 9) ^ ((unsigned int)a2->m128i_i64[0] >> 4)))); ; i = v12 & v16 )
  {
    v15 = v9 + ((unsigned __int64)i << 6);
    if ( v10 == *(_QWORD *)v15 && v11 == *(_QWORD *)(v15 + 8) )
      break;
    if ( unk_4FEE4D0 == *(_QWORD *)v15 && unk_4FEE4D8 == *(_QWORD *)(v15 + 8) )
      goto LABEL_8;
    v16 = v13 + i;
    ++v13;
  }
  v33 = v35;
  v34 = 0x100000000LL;
  if ( !*(_DWORD *)(v15 + 24) )
  {
LABEL_8:
    v33 = v35;
    v34 = 0x300000000LL;
    if ( (unsigned __int8)sub_2526B50(a1, a2, a3, (__int64)&v33, a5, a4, 1u) )
    {
      if ( !(_DWORD)v34 )
      {
        LOBYTE(v30) = 0;
        goto LABEL_13;
      }
      if ( a3 )
      {
        v17 = sub_2554630(a1, a3, a2, &v33);
        if ( v17 )
          goto LABEL_12;
      }
      if ( (unsigned __int8)(sub_2509800(a2) - 2) <= 1u )
      {
        v29 = 0;
        LOBYTE(v30) = 1;
LABEL_13:
        if ( v33 != v35 )
          _libc_free((unsigned __int64)v33);
        return v29;
      }
    }
    v17 = sub_250D070(a2);
LABEL_12:
    v29 = v17;
    LOBYTE(v30) = 1;
    goto LABEL_13;
  }
  v26 = a4;
  sub_2511E10((__int64)&v33, (__int64 *)(v15 + 16), v12, (__int64)a4, v11);
  a4 = v26;
  if ( !(_DWORD)v34 )
  {
    if ( v33 != v35 )
    {
      _libc_free((unsigned __int64)v33);
      a4 = v26;
    }
    goto LABEL_8;
  }
  v19 = *((_QWORD *)v33 + 2) == 0;
  v28 = a3;
  if ( v19 )
    sub_4263D6(v33, (unsigned int)v34, v18);
  v20 = (*((__int64 (__fastcall **)(_BYTE *, __m128i *, __int64 *, _BYTE *))v33 + 3))(v33, a2, &v28, v26);
  v21 = (unsigned __int64)v33;
  v31 = v20;
  v32 = v22;
  v23 = &v33[32 * (unsigned int)v34];
  v29 = v20;
  v30 = v22;
  if ( v33 != v23 )
  {
    do
    {
      v24 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))*((_QWORD *)v23 - 2);
      v23 -= 32;
      if ( v24 )
        v24(v23, v23, 3);
    }
    while ( (_BYTE *)v21 != v23 );
    v23 = v33;
  }
  if ( v23 != v35 )
    _libc_free((unsigned __int64)v23);
  return v29;
}
