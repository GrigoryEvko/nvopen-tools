// Function: sub_2260240
// Address: 0x2260240
//
__int64 __fastcall sub_2260240(_QWORD *a1, __int64 a2)
{
  _BYTE *v2; // r14
  _BYTE *v3; // r13
  __m128i *v4; // rdi
  __int64 (__fastcall *v5)(__int64); // rax
  _BYTE *v6; // rax
  __int64 v7; // rdi
  __int64 v8; // r15
  __m128i *v9; // r15
  __m128i *v10; // r14
  __m128i *v11; // rdi
  __int64 (__fastcall *v12)(__int64); // rax
  __int64 v13; // rdi
  unsigned __int64 v15; // [rsp+8h] [rbp-1C8h]
  unsigned __int64 v16; // [rsp+10h] [rbp-1C0h]
  unsigned __int64 v17; // [rsp+18h] [rbp-1B8h]
  unsigned __int64 v18; // [rsp+20h] [rbp-1B0h]
  unsigned __int64 v19; // [rsp+28h] [rbp-1A8h]
  unsigned __int64 v20; // [rsp+38h] [rbp-198h]
  unsigned int v21; // [rsp+44h] [rbp-18Ch]
  unsigned __int64 v22; // [rsp+48h] [rbp-188h]
  unsigned __int64 v23; // [rsp+58h] [rbp-178h]
  __m128i v24; // [rsp+60h] [rbp-170h] BYREF
  __m128i v25; // [rsp+70h] [rbp-160h]
  __m128i v26; // [rsp+80h] [rbp-150h]
  __m128i v27; // [rsp+90h] [rbp-140h]
  _BYTE v28[16]; // [rsp+A0h] [rbp-130h] BYREF
  __int64 (__fastcall *v29)(__int64); // [rsp+B0h] [rbp-120h]
  __int64 v30; // [rsp+B8h] [rbp-118h]
  __int64 (__fastcall *v31)(__int64); // [rsp+C0h] [rbp-110h]
  __int64 v32; // [rsp+C8h] [rbp-108h]
  __int64 (__fastcall *v33)(__int64 *); // [rsp+D0h] [rbp-100h]
  __int64 v34; // [rsp+D8h] [rbp-F8h]
  _BYTE v35[16]; // [rsp+E0h] [rbp-F0h] BYREF
  __int64 (__fastcall *v36)(__int64); // [rsp+F0h] [rbp-E0h]
  __int64 v37; // [rsp+F8h] [rbp-D8h]
  __int64 (__fastcall *v38)(__int64); // [rsp+100h] [rbp-D0h]
  __int64 v39; // [rsp+108h] [rbp-C8h]
  __int64 (__fastcall *v40)(_QWORD *); // [rsp+110h] [rbp-C0h]
  __int64 v41; // [rsp+118h] [rbp-B8h]
  __m128i v42; // [rsp+120h] [rbp-B0h] BYREF
  __m128i v43; // [rsp+130h] [rbp-A0h] BYREF
  __m128i v44; // [rsp+140h] [rbp-90h] BYREF
  __m128i v45; // [rsp+150h] [rbp-80h] BYREF
  unsigned __int64 v46; // [rsp+160h] [rbp-70h]
  unsigned __int64 v47; // [rsp+168h] [rbp-68h]
  unsigned __int64 v48; // [rsp+170h] [rbp-60h]
  unsigned __int64 v49; // [rsp+178h] [rbp-58h]
  unsigned __int64 v50; // [rsp+180h] [rbp-50h]
  unsigned __int64 v51; // [rsp+188h] [rbp-48h]
  unsigned __int64 v52; // [rsp+190h] [rbp-40h]
  unsigned __int64 v53; // [rsp+198h] [rbp-38h]

  sub_BA9680(&v42, a1);
  v21 = 0;
  v16 = v46;
  v24 = _mm_loadu_si128(&v42);
  v18 = v47;
  v25 = _mm_loadu_si128(&v43);
  v20 = v48;
  v26 = _mm_loadu_si128(&v44);
  v23 = v49;
  v27 = _mm_loadu_si128(&v45);
  v15 = v50;
  v17 = v51;
  v19 = v52;
  v22 = v53;
  while ( 1 )
  {
    if ( *(_OWORD *)&v25 == __PAIR128__(v23, v20)
      && *(_OWORD *)&v24 == __PAIR128__(v18, v16)
      && __PAIR128__(v22, v19) == *(_OWORD *)&v27
      && *(_OWORD *)&v26 == __PAIR128__(v17, v15) )
    {
      return 0;
    }
    v2 = v28;
    v30 = 0;
    v3 = v28;
    v4 = &v24;
    v29 = sub_C11C50;
    v32 = 0;
    v31 = sub_C11C70;
    v34 = 0;
    v33 = sub_C11C90;
    v5 = sub_C11C30;
    if ( ((unsigned __int8)sub_C11C30 & 1) != 0 )
      goto LABEL_4;
    while ( 2 )
    {
      v6 = (_BYTE *)v5((__int64)v4);
      if ( !v6 )
      {
        while ( 1 )
        {
          v2 += 16;
          if ( v35 == v2 )
LABEL_23:
            BUG();
          v7 = *((_QWORD *)v3 + 3);
          v5 = (__int64 (__fastcall *)(__int64))*((_QWORD *)v3 + 2);
          v3 = v2;
          v4 = (__m128i *)((char *)&v24 + v7);
          if ( ((unsigned __int8)v5 & 1) != 0 )
            break;
          v6 = (_BYTE *)v5((__int64)v4);
          if ( v6 )
            goto LABEL_9;
        }
LABEL_4:
        v5 = *(__int64 (__fastcall **)(__int64))((char *)v5 + v4->m128i_i64[0] - 1);
        continue;
      }
      break;
    }
LABEL_9:
    v8 = (__int64)v6;
    if ( !*v6 && !sub_B2FC80((__int64)v6) )
    {
      v21 += sub_225FE40(v8);
      if ( v21 > *(_DWORD *)(a2 + 1440) )
        return 1;
    }
    v9 = (__m128i *)v35;
    v37 = 0;
    v10 = (__m128i *)v35;
    v11 = &v24;
    v39 = 0;
    v36 = sub_C11BA0;
    v41 = 0;
    v38 = sub_C11BD0;
    v40 = sub_C11C00;
    v12 = sub_C11B70;
    if ( ((unsigned __int8)sub_C11B70 & 1) == 0 )
      goto LABEL_13;
LABEL_12:
    v12 = *(__int64 (__fastcall **)(__int64))((char *)v12 + v11->m128i_i64[0] - 1);
LABEL_13:
    while ( !(unsigned __int8)v12((__int64)v11) )
    {
      if ( &v42 == ++v9 )
        goto LABEL_23;
      v13 = v10[1].m128i_i64[1];
      v12 = (__int64 (__fastcall *)(__int64))v10[1].m128i_i64[0];
      v10 = v9;
      v11 = (__m128i *)((char *)&v24 + v13);
      if ( ((unsigned __int8)v12 & 1) != 0 )
        goto LABEL_12;
    }
  }
}
