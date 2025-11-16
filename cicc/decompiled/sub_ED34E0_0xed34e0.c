// Function: sub_ED34E0
// Address: 0xed34e0
//
__int64 *__fastcall sub_ED34E0(__int64 *a1, char *a2, __int64 a3, __int64 a4)
{
  char *v5; // r13
  char *v6; // rax
  unsigned int v7; // ecx
  __int64 v8; // r15
  char v9; // dl
  __int64 v10; // rsi
  __int64 v11; // rsi
  unsigned int v12; // ecx
  char *v13; // rax
  __int64 v14; // r13
  char *v15; // rdx
  char v16; // si
  __int64 v17; // rdi
  __int64 v18; // rdi
  __int64 v19; // rbx
  __int64 v20; // rsi
  unsigned __int64 v21; // rdx
  __m128i *v22; // rdi
  const __m128i *v23; // r15
  __int64 v24; // rbx
  const __m128i *v25; // r12
  bool v26; // zf
  __m128i *v27; // rdi
  char *v29; // [rsp+18h] [rbp-158h]
  __int64 v31; // [rsp+38h] [rbp-138h] BYREF
  __int64 v32; // [rsp+68h] [rbp-108h] BYREF
  __m128i v33; // [rsp+70h] [rbp-100h] BYREF
  __m128i *v34; // [rsp+80h] [rbp-F0h] BYREF
  __int64 v35; // [rsp+88h] [rbp-E8h]
  __m128i v36; // [rsp+90h] [rbp-E0h] BYREF
  _BYTE *v37; // [rsp+A0h] [rbp-D0h] BYREF
  __int64 v38; // [rsp+A8h] [rbp-C8h]
  __int64 v39; // [rsp+B0h] [rbp-C0h]
  _BYTE v40[184]; // [rsp+B8h] [rbp-B8h] BYREF

  v29 = &a2[a3];
  if ( &a2[a3] > a2 )
  {
    v5 = a2;
    do
    {
      v6 = v5;
      v7 = 0;
      v8 = 0;
      while ( v6 )
      {
        v9 = *v6;
        v10 = *v6 & 0x7F;
        if ( v7 > 0x3E )
        {
          if ( v7 == 63 )
          {
            if ( v10 != (v9 & 1) )
              break;
          }
          else if ( (*v6 & 0x7F) != 0 )
          {
            break;
          }
        }
        v11 = v10 << v7;
        ++v6;
        v7 += 7;
        v8 += v11;
        if ( v9 >= 0 )
          goto LABEL_8;
      }
      v8 = 0;
LABEL_8:
      v12 = 0;
      v13 = &v5[(_DWORD)v6 - (_DWORD)v5];
      v14 = 0;
      v15 = v13;
      do
      {
        if ( !v15 )
        {
LABEL_12:
          v38 = 0;
          v39 = 128;
          v19 = (__int64)&v13[(_DWORD)v15 - (_DWORD)v13];
          v37 = v40;
          goto LABEL_13;
        }
        v16 = *v15;
        v17 = *v15 & 0x7F;
        if ( v12 > 0x3E )
        {
          if ( v12 == 63 )
          {
            if ( v17 != (v16 & 1) )
              goto LABEL_12;
          }
          else if ( (*v15 & 0x7F) != 0 )
          {
            goto LABEL_12;
          }
        }
        v18 = v17 << v12;
        ++v15;
        v12 += 7;
        v14 += v18;
      }
      while ( v16 < 0 );
      v38 = 0;
      v39 = 128;
      v19 = (__int64)&v13[(_DWORD)v15 - (_DWORD)v13];
      v33 = 0u;
      v37 = v40;
      if ( v14 )
      {
        if ( (unsigned __int8)sub_C5E690() )
          sub_40930A(&v31, v19, v14, &v37, v8);
        v20 = 23;
        sub_ED07D0(a1, 23);
LABEL_19:
        if ( v37 != v40 )
          _libc_free(v37, v20);
        return a1;
      }
LABEL_13:
      v33.m128i_i64[0] = v19;
      v5 = (char *)(v19 + v8);
      v33.m128i_i64[1] = v8;
      v20 = (__int64)&v34;
      v35 = 0;
      v34 = &v36;
      sub_C937F0(&v33, (__int64)&v34, &unk_3F871B2, 1u, 0xFFFFFFFFLL, 1);
      v22 = v34;
      v23 = v34;
      if ( &v34[(unsigned int)v35] != v34 )
      {
        v24 = a4;
        v25 = &v34[(unsigned int)v35];
        while ( 1 )
        {
          v26 = *(_QWORD *)(v24 + 16) == 0;
          v36 = _mm_loadu_si128(v23);
          if ( v26 )
            sub_4263D6(v22, v20, v21);
          v20 = v24;
          v22 = (__m128i *)&v32;
          (*(void (__fastcall **)(__int64 *, __int64, __m128i *))(v24 + 24))(&v32, v24, &v36);
          v21 = v32 & 0xFFFFFFFFFFFFFFFELL;
          if ( (v32 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            break;
          if ( v25 == ++v23 )
          {
            a4 = v24;
            v22 = v34;
            goto LABEL_24;
          }
        }
        v27 = v34;
        *a1 = v21 | 1;
        if ( v27 != &v36 )
          _libc_free(v27, v24);
        goto LABEL_19;
      }
LABEL_24:
      if ( v5 < v29 )
      {
        do
        {
          if ( *v5 )
            break;
          ++v5;
        }
        while ( v29 != v5 );
      }
      if ( v22 != &v36 )
        _libc_free(v22, v20);
      if ( v37 != v40 )
        _libc_free(v37, v20);
    }
    while ( v29 > v5 );
  }
  v37 = 0;
  *a1 = 1;
  sub_9C66B0((__int64 *)&v37);
  return a1;
}
