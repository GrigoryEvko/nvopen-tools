// Function: sub_256F690
// Address: 0x256f690
//
__int64 __fastcall sub_256F690(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v6; // r11
  char *v7; // r12
  bool v8; // zf
  const __m128i *v9; // rbx
  const __m128i *i; // r14
  __int64 v11; // rax
  __m128i v12; // xmm0
  unsigned __int64 v13; // rdx
  unsigned __int8 *v14; // rbx
  unsigned int v15; // r12d
  __int64 v17; // rax
  unsigned __int64 v18; // rax
  unsigned __int8 *v19; // rcx
  __int64 v20; // rbx
  char v21; // al
  __m128i v22; // [rsp+0h] [rbp-B0h] BYREF
  unsigned __int8 *v23; // [rsp+18h] [rbp-98h]
  __int64 v24; // [rsp+20h] [rbp-90h]
  _BYTE **v25; // [rsp+28h] [rbp-88h]
  unsigned __int8 *v26[2]; // [rsp+30h] [rbp-80h] BYREF
  _BYTE *v27; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v28; // [rsp+48h] [rbp-68h]
  unsigned int v29; // [rsp+4Ch] [rbp-64h]
  _BYTE v30[96]; // [rsp+50h] [rbp-60h] BYREF

  v6 = 2;
  v7 = (char *)&unk_438A62D;
  v24 = a2;
  v27 = v30;
  v29 = 3;
  v25 = &v27;
  while ( 1 )
  {
    v8 = *(_BYTE *)(a1 + 105) == 0;
    v28 = 0;
    if ( !v8 )
    {
      v9 = *(const __m128i **)(a1 + 144);
      for ( i = (const __m128i *)((char *)v9 + 24 * *(unsigned int *)(a1 + 152));
            i != v9;
            v9 = (const __m128i *)((char *)v9 + 24) )
      {
        if ( (v9[1].m128i_i8[0] & (unsigned __int8)v6) != 0 )
        {
          v11 = v28;
          v12 = _mm_loadu_si128(v9);
          v13 = v28 + 1LL;
          if ( v13 > v29 )
          {
            LOBYTE(v23) = v6;
            v22 = v12;
            sub_C8D5F0((__int64)v25, v30, v13, 0x10u, a5, a6);
            v11 = v28;
            v12 = _mm_load_si128(&v22);
            v6 = (char)v23;
          }
          *(__m128i *)&v27[16 * v11] = v12;
          ++v28;
        }
      }
      v14 = (unsigned __int8 *)sub_250D070((_QWORD *)(a1 + 72));
      if ( (unsigned int)*v14 - 12 > 1 )
      {
        v17 = sub_2554630(v24, a1, (__int64 *)(a1 + 72), (__int64)v25);
        if ( v17 )
        {
          v23 = (unsigned __int8 *)v17;
          if ( v14 != (unsigned __int8 *)v17 )
          {
            v18 = sub_2509740((_QWORD *)(a1 + 72));
            v19 = v23;
            if ( !v18
              || (v20 = *(_QWORD *)(v24 + 208),
                  v26[1] = (unsigned __int8 *)sub_2509740((_QWORD *)(a1 + 72)),
                  v26[0] = v23,
                  v21 = sub_250C1E0(v26, v20),
                  v19 = v23,
                  v21) )
            {
              if ( (unsigned __int8)sub_256F570(v24, *(_QWORD *)(a1 + 72), *(_QWORD *)(a1 + 80), v19, 1u) )
                break;
            }
          }
        }
      }
    }
    if ( &unk_438A62F == (_UNKNOWN *)++v7 )
    {
      v15 = 1;
      goto LABEL_13;
    }
    v6 = *v7;
  }
  v15 = 0;
LABEL_13:
  if ( v27 != v30 )
    _libc_free((unsigned __int64)v27);
  return v15;
}
