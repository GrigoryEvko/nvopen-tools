// Function: sub_7A2650
// Address: 0x7a2650
//
__int64 __fastcall sub_7A2650(__int64 a1, int a2, FILE *a3, __int64 a4, __m128i *a5, unsigned int a6)
{
  __int64 result; // rax
  int v11; // ecx
  char v12; // dl
  unsigned int v13; // edx
  unsigned int v14; // eax
  size_t v15; // rdx
  __int64 v16; // rax
  char *v17; // rcx
  __m128i *v18; // r10
  int v19; // eax
  __m128i *v20; // r10
  __int64 v21; // rax
  __int64 v22; // rax
  int v23; // eax
  _QWORD *v24; // rdi
  __m128i *v25; // r10
  __int64 v26; // rax
  unsigned int v27; // ecx
  __int64 v28; // rsi
  unsigned __int64 v29; // r14
  unsigned int v30; // edx
  __m128i *v31; // rax
  __m128i v32; // xmm0
  __m128i *v33; // rax
  __int64 v34; // r8
  int v35; // esi
  __int64 v36; // rcx
  unsigned int v37; // edx
  __m128i **i; // rax
  size_t v39; // [rsp+0h] [rbp-150h]
  size_t v40; // [rsp+0h] [rbp-150h]
  int v41; // [rsp+Ch] [rbp-144h]
  unsigned int v42; // [rsp+Ch] [rbp-144h]
  __int64 v43; // [rsp+10h] [rbp-140h]
  __m128i *v44; // [rsp+10h] [rbp-140h]
  unsigned __int64 v45; // [rsp+18h] [rbp-138h]
  unsigned __int64 v47; // [rsp+20h] [rbp-130h]
  __m128i *v49; // [rsp+28h] [rbp-128h]
  unsigned int v50; // [rsp+34h] [rbp-11Ch] BYREF
  _QWORD *v51; // [rsp+38h] [rbp-118h] BYREF
  __int64 *v52; // [rsp+40h] [rbp-110h] BYREF
  __int128 v53; // [rsp+48h] [rbp-108h]
  __int64 *v54; // [rsp+58h] [rbp-F8h]
  __int64 v55; // [rsp+60h] [rbp-F0h] BYREF
  unsigned int v56; // [rsp+68h] [rbp-E8h]
  int v57; // [rsp+6Ch] [rbp-E4h]
  void *s; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v59; // [rsp+78h] [rbp-D8h]
  __int64 v60; // [rsp+80h] [rbp-D0h]
  int v61; // [rsp+88h] [rbp-C8h]
  __int64 v62; // [rsp+90h] [rbp-C0h]
  __m128i v63; // [rsp+C0h] [rbp-90h] BYREF
  __int64 v64; // [rsp+D0h] [rbp-80h]
  char v65; // [rsp+E4h] [rbp-6Ch]
  char v66; // [rsp+E5h] [rbp-6Bh]
  __int64 v67; // [rsp+118h] [rbp-38h]

  v50 = 1;
  result = dword_4F07588;
  if ( dword_4F07588 )
  {
    result = 0;
    if ( !dword_4D03F94 )
    {
      if ( sub_730800(a1) )
      {
        sub_72C970(a4);
        return v50;
      }
      if ( dword_4F08058 )
      {
        sub_771BE0(a1, (unsigned int)dword_4F08058);
        dword_4F08058 = 0;
      }
      sub_774A30((__int64)&v55, a2);
      if ( a2 )
        v66 |= 0x10u;
      v64 = *(_QWORD *)&a3->_flags;
      if ( (*(_BYTE *)(a1 + 72) & 1) != 0 )
      {
        sub_6855B0(0xBA8u, a3, &v63);
        v50 = 0;
      }
      else
      {
        v11 = 16;
        v12 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 56) + 40LL) + 32LL) + 140LL);
        v45 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 56) + 40LL) + 32LL);
        if ( (unsigned __int8)(v12 - 2) <= 1u )
        {
LABEL_11:
          if ( (*(_BYTE *)(a1 + 49) & 1) != 0 && (unsigned __int8)(v12 - 9) <= 2u )
          {
            if ( (*(_BYTE *)(*(_QWORD *)(v45 + 168) + 110LL) & 1) != 0 )
              v66 |= 1u;
          }
          else if ( (unsigned __int8)(v12 - 8) > 3u )
          {
            v43 = 16;
            v15 = 8;
            v14 = 16;
            goto LABEL_17;
          }
          v13 = (unsigned int)(v11 + 7) >> 3;
          v14 = v13 + 9;
          if ( (((_BYTE)v13 + 9) & 7) != 0 )
            v14 = v13 + 17 - (((_BYTE)v13 + 9) & 7);
          v43 = v14;
          v15 = v14 - 8LL;
LABEL_17:
          v16 = v11 + v14;
          if ( (unsigned int)v16 > 0x400 )
          {
            v39 = v15;
            v41 = v16 + 16;
            v22 = sub_822B10((unsigned int)(v16 + 16));
            v15 = v39;
            *(_QWORD *)v22 = v60;
            *(_DWORD *)(v22 + 8) = v41;
            *(_DWORD *)(v22 + 12) = v61;
            v17 = (char *)(v22 + 16);
            v60 = v22;
          }
          else
          {
            if ( (v16 & 7) != 0 )
              v16 = (_DWORD)v16 + 8 - (unsigned int)(v16 & 7);
            v17 = (char *)s;
            if ( 0x10000 - ((int)s - (int)v59) < (unsigned int)v16 )
            {
              v40 = v15;
              v42 = v16;
              sub_772E70(&s);
              v17 = (char *)s;
              v15 = v40;
              v16 = v42;
            }
            s = &v17[v16];
          }
          v18 = (__m128i *)((char *)memset(v17, 0, v15) + v43);
          v18[-1].m128i_i64[1] = v45;
          if ( (unsigned __int8)(*(_BYTE *)(v45 + 140) - 9) <= 2u )
            v18->m128i_i64[0] = 0;
          v52 = (__int64 *)v18;
          v54 = (__int64 *)v18;
          v44 = v18;
          v53 = 0;
          DWORD1(v53) = 1;
          v19 = sub_799B70((__int64)&v55, a1, a3, (const __m128i *)&v52, 0, a6);
          v20 = v44;
          if ( !v19 )
          {
            if ( (v65 & 0x40) == 0 )
            {
LABEL_27:
              v50 = 0;
              v21 = v62;
              goto LABEL_28;
            }
            sub_72C970(a4);
            goto LABEL_42;
          }
          v21 = v62;
          if ( v62 )
          {
            if ( !a2 )
            {
LABEL_38:
              v50 = 0;
LABEL_28:
              if ( !v21 )
              {
LABEL_34:
                *a5 = _mm_loadu_si128(&v63);
                sub_771990((__int64)&v55);
                return v50;
              }
LABEL_29:
              if ( !a2 || !(unsigned int)sub_799890((__int64)&v55) )
              {
LABEL_30:
                v50 = 0;
                goto LABEL_34;
              }
LABEL_43:
              if ( !v50 || !v67 )
                goto LABEL_34;
              sub_773640((__int64)&v55);
              goto LABEL_30;
            }
            v23 = sub_799890((__int64)&v55);
            v20 = v44;
            if ( !v23 )
            {
              v21 = v62;
              goto LABEL_38;
            }
          }
          if ( v67 )
          {
            sub_773640((__int64)&v55);
            goto LABEL_27;
          }
          v47 = (unsigned __int64)v20;
          v51 = sub_724DC0();
          sub_724C70((__int64)v51, 6);
          v24 = v51;
          v25 = (__m128i *)v47;
          *((_BYTE *)v51 + 176) = 1;
          v26 = *(_QWORD *)(a1 + 8);
          if ( v26 )
            v24[23] = v26;
          v27 = v56;
          v28 = v55;
          v29 = v47 >> 3;
          v30 = (v47 >> 3) & v56;
          v31 = (__m128i *)(v55 + 16LL * v30);
          if ( v31->m128i_i64[0] )
          {
            v32 = _mm_loadu_si128(v31);
            v31->m128i_i64[0] = v47;
            v31->m128i_i64[1] = (__int64)v24;
            do
            {
              v30 = v27 & (v30 + 1);
              v33 = (__m128i *)(v28 + 16LL * v30);
            }
            while ( v33->m128i_i64[0] );
            *v33 = v32;
          }
          else
          {
            v31->m128i_i64[0] = v47;
            v31->m128i_i64[1] = (__int64)v51;
          }
          ++v57;
          if ( 2 * v57 > v27 )
          {
            sub_7704A0((__int64)&v55);
            v25 = (__m128i *)v47;
          }
          v34 = a4;
          v49 = v25;
          if ( !sub_77D750((__int64)&v55, v25, (__int64)v25, v45, v34) )
            v50 = 0;
          v35 = v56;
          v36 = v55;
          v37 = v56 & v29;
          for ( i = (__m128i **)(v55 + 16LL * (v56 & (unsigned int)v29)); *i != v49; i = (__m128i **)(v55 + 16LL * v37) )
            v37 = v56 & (v37 + 1);
          *i = 0;
          if ( *(_QWORD *)(v36 + 16LL * ((v37 + 1) & v35)) )
            sub_771200(v55, v56, v37);
          --v57;
          sub_724E30((__int64)&v51);
LABEL_42:
          if ( !v62 )
            goto LABEL_43;
          goto LABEL_29;
        }
        v11 = sub_7764B0((__int64)&v55, v45, &v50);
        if ( v50 )
        {
          v12 = *(_BYTE *)(v45 + 140);
          goto LABEL_11;
        }
      }
      if ( (v65 & 0x40) != 0 )
      {
        sub_72C970(a4);
        v50 = 1;
      }
      goto LABEL_34;
    }
  }
  return result;
}
