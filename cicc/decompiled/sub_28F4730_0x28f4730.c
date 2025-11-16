// Function: sub_28F4730
// Address: 0x28f4730
//
_BYTE *__fastcall sub_28F4730(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // r12
  _BYTE *v4; // r15
  unsigned int v5; // ecx
  char *v7; // rsi
  unsigned int v9; // r10d
  unsigned int v10; // eax
  __int64 v11; // r9
  unsigned int v12; // edi
  char *v13; // rdx
  __int64 v14; // r8
  __int64 v15; // rdi
  unsigned __int64 v16; // rbx
  unsigned int i; // r12d
  __int64 v18; // r9
  __int64 v19; // rax
  __int64 v20; // r10
  int v21; // edx
  unsigned int v22; // r11d
  __int64 v23; // r15
  unsigned __int64 v24; // rax
  __int64 *v25; // rdi
  __int64 v26; // r15
  __int64 v27; // rax
  char *v28; // r9
  char *v29; // rcx
  size_t v30; // rdx
  char *v31; // rax
  __int64 v32; // rdi
  __m128i *v33; // r14
  __int64 v34; // rbx
  __m128i *v35; // r12
  __int64 v36; // rbx
  __int64 v37; // r15
  __int64 v38; // rax
  char *v39; // r10
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rcx
  __m128i v43; // xmm0
  __int64 v44; // rax
  unsigned __int64 v45; // r10
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rsi
  __int64 v49; // r8
  __int64 v50; // r9
  unsigned __int64 v51; // rax
  int v52; // ecx
  _QWORD *v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // r9
  int v57; // eax
  unsigned int v58; // eax
  __int64 v59; // rcx
  char *v60; // r10
  __int64 v61; // rbx
  __m128i *v62; // rsi
  __int64 v63; // r8
  __int64 v64; // r9
  char *v65; // r10
  __int64 v66; // r11
  __m128i *v67; // r12
  unsigned __int64 v68; // rdx
  unsigned __int64 v69; // rax
  const __m128i *v70; // rax
  unsigned __int64 v72; // [rsp+0h] [rbp-150h]
  __int64 v73; // [rsp+0h] [rbp-150h]
  __int64 v74; // [rsp+8h] [rbp-148h]
  __int64 v75; // [rsp+10h] [rbp-140h]
  char *v77; // [rsp+18h] [rbp-138h]
  size_t v78; // [rsp+28h] [rbp-128h]
  char *v79; // [rsp+28h] [rbp-128h]
  __int64 v80; // [rsp+28h] [rbp-128h]
  __int64 v81[2]; // [rsp+30h] [rbp-120h] BYREF
  void *src; // [rsp+40h] [rbp-110h] BYREF
  __int64 v83; // [rsp+48h] [rbp-108h]
  _BYTE v84[64]; // [rsp+50h] [rbp-100h] BYREF
  _BYTE *v85; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v86; // [rsp+98h] [rbp-B8h]
  _BYTE v87[32]; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v88; // [rsp+C0h] [rbp-90h]
  __int64 v89; // [rsp+C8h] [rbp-88h]
  __int16 v90; // [rsp+D0h] [rbp-80h]
  __int64 v91; // [rsp+D8h] [rbp-78h]
  void **v92; // [rsp+E0h] [rbp-70h]
  void **v93; // [rsp+E8h] [rbp-68h]
  __int64 v94; // [rsp+F0h] [rbp-60h]
  int v95; // [rsp+F8h] [rbp-58h]
  __int16 v96; // [rsp+FCh] [rbp-54h]
  char v97; // [rsp+FEh] [rbp-52h]
  __int64 v98; // [rsp+100h] [rbp-50h]
  __int64 v99; // [rsp+108h] [rbp-48h]
  void *v100; // [rsp+110h] [rbp-40h] BYREF
  void *v101; // [rsp+118h] [rbp-38h] BYREF

  v4 = 0;
  v5 = *(_DWORD *)(a3 + 8);
  if ( v5 > 3 )
  {
    v7 = *(char **)a3;
    v9 = 0;
    src = v84;
    v83 = 0x400000000LL;
    v10 = 1;
    do
    {
      v11 = *(_QWORD *)&v7[16 * v10 - 8];
      if ( v5 <= v10 )
      {
        v14 = v10;
      }
      else
      {
        v12 = 1;
        v13 = &v7[16 * v10];
        while ( 1 )
        {
          v14 = v10++;
          if ( v11 != *((_QWORD *)v13 + 1) )
            break;
          ++v12;
          v13 += 16;
          if ( v5 == v10 )
          {
            v14 = v5;
            break;
          }
        }
        if ( v12 > 1 )
          v9 += v12;
      }
      v10 = v14 + 1;
    }
    while ( v5 > (int)v14 + 1 );
    v4 = 0;
    if ( v9 > 3 )
    {
      v74 = a2;
      v15 = 0;
      v16 = v3;
      for ( i = 1; i < v5; ++i )
      {
        v18 = *(_QWORD *)&v7[16 * i - 8];
        v19 = i;
        if ( v5 <= i )
        {
LABEL_60:
          v5 = *(_DWORD *)(a3 + 8);
        }
        else
        {
          v20 = i + 1;
          v21 = 1;
          v22 = v5 + 1 - i;
          while ( 1 )
          {
            i = v20 - 1;
            if ( v18 != *(_QWORD *)&v7[16 * v19 + 8] )
            {
              if ( v21 == 1 )
                goto LABEL_60;
              goto LABEL_18;
            }
            ++v21;
            v19 = v20;
            if ( v21 == v22 )
              break;
            ++v20;
          }
          i = v20;
          if ( v21 == 1 )
            goto LABEL_60;
LABEL_18:
          v23 = v21 & 0xFFFFFFFE;
          i -= v23;
          v24 = v23 | v16 & 0xFFFFFFFF00000000LL;
          v16 = v24;
          if ( v15 + 1 > (unsigned __int64)HIDWORD(v83) )
          {
            v72 = v24;
            v80 = v18;
            sub_C8D5F0((__int64)&src, v84, v15 + 1, 0x10u, v14, v18);
            v15 = (unsigned int)v83;
            v24 = v72;
            v18 = v80;
          }
          v25 = (__int64 *)((char *)src + 16 * v15);
          *v25 = v18;
          v25[1] = v24;
          v26 = 16 * (i + v23);
          v7 = *(char **)a3;
          v15 = (unsigned int)(v83 + 1);
          v27 = *(unsigned int *)(a3 + 8);
          v28 = (char *)(*(_QWORD *)a3 + v26);
          LODWORD(v83) = v83 + 1;
          v29 = &v7[16 * i];
          v30 = 16 * v27 - v26;
          if ( v28 != &v7[16 * v27] )
          {
            v78 = 16 * v27 - v26;
            v31 = (char *)memmove(&v7[16 * i], v28, v30);
            v7 = *(char **)a3;
            v15 = (unsigned int)v83;
            v30 = v78;
            v29 = v31;
          }
          *(_DWORD *)(a3 + 8) = (&v29[v30] - v7) >> 4;
          v5 = (&v29[v30] - v7) >> 4;
        }
      }
      v32 = 16 * v15;
      v33 = (__m128i *)src;
      v34 = v74;
      v35 = (__m128i *)((char *)src + v32);
      if ( v32 )
      {
        v36 = v32 >> 4;
        while ( 1 )
        {
          v37 = 16 * v36;
          v38 = sub_2207800(16 * v36);
          v39 = (char *)v38;
          if ( v38 )
            break;
          v36 >>= 1;
          if ( !v36 )
          {
            v34 = v74;
            goto LABEL_71;
          }
        }
        v40 = v38 + v37;
        v41 = v38 + 16;
        v42 = v36;
        v34 = v74;
        *(__m128i *)(v41 - 16) = _mm_loadu_si128(v33);
        if ( v40 == v41 )
        {
          v44 = (__int64)v39;
        }
        else
        {
          do
          {
            v43 = _mm_loadu_si128((const __m128i *)(v41 - 16));
            v41 += 16;
            *(__m128i *)(v41 - 16) = v43;
          }
          while ( v40 != v41 );
          v44 = (__int64)&v39[v37 - 16];
        }
        v79 = v39;
        v33->m128i_i64[0] = *(_QWORD *)v44;
        v33->m128i_i32[2] = *(_DWORD *)(v44 + 8);
        sub_28ECAD0(v33, v35, v39, v42);
        v45 = (unsigned __int64)v79;
      }
      else
      {
LABEL_71:
        sub_28EB210(v33, v35);
        v45 = 0;
      }
      j_j___libc_free_0(v45);
      v46 = sub_BD5C60(v34);
      v93 = &v101;
      v91 = v46;
      v92 = &v100;
      v85 = v87;
      v100 = &unk_49DA100;
      v96 = 512;
      v86 = 0x200000000LL;
      v101 = &unk_49DA0B0;
      v47 = *(_QWORD *)(v34 + 40);
      v94 = 0;
      v88 = v47;
      v95 = 0;
      v97 = 7;
      v98 = 0;
      v99 = 0;
      v89 = v34 + 24;
      v90 = 0;
      v48 = *(_QWORD *)sub_B46C60(v34);
      v81[0] = v48;
      if ( v48 && (sub_B96E90((__int64)v81, v48, 1), (v50 = v81[0]) != 0) )
      {
        v51 = (unsigned __int64)v85;
        v52 = v86;
        v53 = &v85[16 * (unsigned int)v86];
        if ( v85 != (_BYTE *)v53 )
        {
          while ( *(_DWORD *)v51 )
          {
            v51 += 16LL;
            if ( v53 == (_QWORD *)v51 )
              goto LABEL_65;
          }
          *(_QWORD *)(v51 + 8) = v81[0];
LABEL_38:
          sub_B91220((__int64)v81, v50);
LABEL_39:
          if ( (unsigned __int8)sub_920620(v34) )
          {
            v57 = *(_BYTE *)(v34 + 1) >> 1;
            if ( v57 == 127 )
              v57 = -1;
            v95 = v57;
          }
          v4 = (_BYTE *)sub_28F4220(a1, (__int64)&v85, (__int64)&src, v54, v55, v56);
          if ( *(_DWORD *)(a3 + 8) )
          {
            v58 = sub_28EF780(a1, v4);
            v59 = *(unsigned int *)(a3 + 8);
            v60 = *(char **)a3;
            v61 = v58;
            v81[1] = (__int64)v4;
            v75 = v59;
            LODWORD(v81[0]) = v58;
            v62 = (__m128i *)&v60[16 * v59];
            v67 = (__m128i *)sub_28EA1A0(v60, (__int64)v62, v81);
            v68 = v75 + 1;
            v69 = *(unsigned int *)(a3 + 12);
            if ( v62 == v67 )
            {
              if ( v68 > v69 )
              {
                sub_C8D5F0(a3, (const void *)(a3 + 16), v68, 0x10u, v63, v64);
                v67 = (__m128i *)(*(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8));
              }
              v67->m128i_i64[0] = v61;
              v67->m128i_i64[1] = (__int64)v4;
              ++*(_DWORD *)(a3 + 8);
            }
            else
            {
              if ( v68 > v69 )
              {
                v77 = v65;
                sub_C8D5F0(a3, (const void *)(a3 + 16), v68, 0x10u, v63, v64);
                v66 = *(unsigned int *)(a3 + 8);
                v64 = 16 * v66;
                v67 = (__m128i *)(*(_QWORD *)a3 + (char *)v67 - v77);
                v62 = (__m128i *)(*(_QWORD *)a3 + 16 * v66);
                v65 = *(char **)a3;
              }
              v70 = (const __m128i *)&v65[v64 - 16];
              if ( v62 )
              {
                *v62 = _mm_loadu_si128(v70);
                v65 = *(char **)a3;
                v66 = *(unsigned int *)(a3 + 8);
                v64 = 16 * v66;
                v70 = (const __m128i *)(*(_QWORD *)a3 + 16 * v66 - 16);
              }
              if ( v67 != v70 )
              {
                memmove(&v65[v64 - ((char *)v70 - (char *)v67)], v67, (char *)v70 - (char *)v67);
                LODWORD(v66) = *(_DWORD *)(a3 + 8);
              }
              *(_DWORD *)(a3 + 8) = v66 + 1;
              v67->m128i_i32[0] = v61;
              v67->m128i_i64[1] = (__int64)v4;
            }
            v4 = 0;
          }
          nullsub_61();
          v100 = &unk_49DA100;
          nullsub_63();
          if ( v85 != v87 )
            _libc_free((unsigned __int64)v85);
          if ( src != v84 )
            _libc_free((unsigned __int64)src);
          return v4;
        }
LABEL_65:
        if ( (unsigned int)v86 >= (unsigned __int64)HIDWORD(v86) )
        {
          if ( HIDWORD(v86) < (unsigned __int64)(unsigned int)v86 + 1 )
          {
            v73 = v81[0];
            sub_C8D5F0((__int64)&v85, v87, (unsigned int)v86 + 1LL, 0x10u, v49, v81[0]);
            v50 = v73;
            v53 = &v85[16 * (unsigned int)v86];
          }
          *v53 = 0;
          v53[1] = v50;
          v50 = v81[0];
          LODWORD(v86) = v86 + 1;
        }
        else
        {
          if ( v53 )
          {
            *(_DWORD *)v53 = 0;
            v53[1] = v50;
            v52 = v86;
            v50 = v81[0];
          }
          LODWORD(v86) = v52 + 1;
        }
      }
      else
      {
        sub_93FB40((__int64)&v85, 0);
        v50 = v81[0];
      }
      if ( !v50 )
        goto LABEL_39;
      goto LABEL_38;
    }
  }
  return v4;
}
