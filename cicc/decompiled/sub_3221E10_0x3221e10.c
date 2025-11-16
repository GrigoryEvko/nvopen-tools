// Function: sub_3221E10
// Address: 0x3221e10
//
void __fastcall sub_3221E10(__int64 a1, char a2, __int64 a3, __int64 a4, __int64 a5, __int64 **a6)
{
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // r8
  void (*v13)(); // rax
  __int64 v14; // rdi
  void (*v15)(); // rax
  __int64 v16; // r15
  __int64 v17; // rdi
  void (*v18)(); // rax
  int v19; // r13d
  unsigned __int16 v20; // ax
  int v21; // ecx
  __int64 *v22; // rsi
  __int64 v23; // rax
  __int64 *v24; // rax
  __int64 v25; // rdx
  __int64 *v26; // r14
  __int64 *v27; // r15
  __m128i *v28; // rdi
  unsigned __int64 v29; // rcx
  unsigned int v30; // eax
  __int64 *v31; // r8
  __int64 v32; // rdx
  __int64 v33; // r9
  __m128i *v34; // rsi
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 *v37; // rdx
  __int64 v38; // rdi
  __int64 v39; // r8
  void (*v40)(); // rax
  __int64 v41; // r14
  __m128i *v42; // r13
  unsigned __int64 v43; // rax
  __m128i *v44; // r14
  __m128i *v45; // rdi
  unsigned __int64 v46; // r15
  __int64 v47; // rdi
  void (*v48)(); // rax
  __int64 v49; // rsi
  __int64 v50; // rdi
  __int64 v51; // r9
  void (*v52)(); // rax
  __int64 v53; // rsi
  _BOOL4 v54; // edi
  int v55; // r9d
  int v56; // r11d
  char *v57; // rax
  __int64 v58; // rdx
  __m128i v59; // rax
  char v60; // al
  int v61; // r9d
  _QWORD *v62; // rdx
  char v63; // al
  __m128i *v64; // rsi
  char v65; // dl
  const char **v66; // rdi
  __m128i v67; // xmm0
  __m128i v68; // xmm1
  __int64 v69; // rax
  __m128i *v70; // r13
  __m128i *v71; // rax
  __m128i v72; // xmm4
  unsigned int v73; // ecx
  __int64 v74; // rax
  char *v75; // r13
  __int64 v76; // [rsp+8h] [rbp-198h]
  __int64 v77; // [rsp+18h] [rbp-188h]
  __int64 v78; // [rsp+20h] [rbp-180h]
  __int64 v79; // [rsp+30h] [rbp-170h]
  int v80; // [rsp+3Ch] [rbp-164h]
  __int64 v81; // [rsp+40h] [rbp-160h]
  int v82; // [rsp+48h] [rbp-158h]
  __int64 v83; // [rsp+48h] [rbp-158h]
  void (*v84)(); // [rsp+48h] [rbp-158h]
  __int64 v85; // [rsp+50h] [rbp-150h]
  int v86; // [rsp+60h] [rbp-140h]
  int v88; // [rsp+68h] [rbp-138h]
  __m128i *v89; // [rsp+68h] [rbp-138h]
  __m128i *v90; // [rsp+70h] [rbp-130h] BYREF
  __int64 v91; // [rsp+78h] [rbp-128h]
  __m128i v92; // [rsp+80h] [rbp-120h] BYREF
  __m128i v93; // [rsp+90h] [rbp-110h] BYREF
  __int64 v94; // [rsp+A0h] [rbp-100h]
  _QWORD v95[4]; // [rsp+B0h] [rbp-F0h] BYREF
  char v96; // [rsp+D0h] [rbp-D0h]
  char v97; // [rsp+D1h] [rbp-CFh]
  __m128i v98; // [rsp+E0h] [rbp-C0h] BYREF
  __m128i v99; // [rsp+F0h] [rbp-B0h] BYREF
  __int64 v100; // [rsp+100h] [rbp-A0h]
  const char *v101; // [rsp+110h] [rbp-90h] BYREF
  __int64 v102; // [rsp+118h] [rbp-88h]
  __int64 v103; // [rsp+120h] [rbp-80h]
  __int64 v104; // [rsp+128h] [rbp-78h]
  __int16 v105; // [rsp+130h] [rbp-70h]
  __m128i v106; // [rsp+140h] [rbp-60h] BYREF
  __m128i v107; // [rsp+150h] [rbp-50h]
  __int64 v108; // [rsp+160h] [rbp-40h]

  v8 = *(_QWORD *)(a1 + 8);
  v103 = a3;
  v9 = *(_QWORD *)(a5 + 408);
  v99.m128i_i64[0] = a3;
  LOWORD(v100) = 1283;
  if ( v9 )
    a5 = v9;
  v105 = 1283;
  v101 = "Length of Public ";
  v106.m128i_i64[0] = (__int64)&v101;
  v107.m128i_i64[0] = (__int64)" Info";
  v98.m128i_i64[0] = (__int64)"pub";
  v85 = a5;
  v104 = a4;
  LOWORD(v108) = 770;
  v99.m128i_i64[1] = a4;
  v10 = sub_31F0F50(v8);
  v11 = *(_QWORD *)(a1 + 8);
  v76 = v10;
  v12 = *(_QWORD *)(v11 + 224);
  v13 = *(void (**)())(*(_QWORD *)v12 + 120LL);
  v106.m128i_i64[0] = (__int64)"DWARF Version";
  LOWORD(v108) = 259;
  if ( v13 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, __m128i *, __int64))v13)(v12, &v106, 1);
    v11 = *(_QWORD *)(a1 + 8);
  }
  sub_31DC9F0(v11, 2);
  v14 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL);
  v15 = *(void (**)())(*(_QWORD *)v14 + 120LL);
  v106.m128i_i64[0] = (__int64)"Offset of Compilation Unit Info";
  LOWORD(v108) = 259;
  if ( v15 != nullsub_98 )
    ((void (__fastcall *)(__int64, __m128i *, __int64))v15)(v14, &v106, 1);
  sub_321F970(a1, (_QWORD *)v85);
  v16 = *(_QWORD *)(a1 + 8);
  v17 = *(_QWORD *)(v16 + 224);
  v18 = *(void (**)())(*(_QWORD *)v17 + 120LL);
  v106.m128i_i64[0] = (__int64)"Compilation Unit Length";
  LOWORD(v108) = 259;
  if ( v18 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, __m128i *, __int64))v18)(v17, &v106, 1);
    v16 = *(_QWORD *)(a1 + 8);
  }
  v19 = 0;
  v88 = sub_31DF740(*(_QWORD *)(v85 + 184));
  if ( (unsigned __int16)sub_3220AA0(*(_QWORD *)(v85 + 208)) > 4u )
    v19 = 8 * (*(_BYTE *)(*(_QWORD *)(v85 + 208) + 3769LL) != 0);
  v82 = sub_31DF6B0(*(_QWORD *)(v85 + 184));
  v20 = sub_3220AA0(*(_QWORD *)(v85 + 208));
  sub_31F0F00(v16, v19 + v88 + *(_DWORD *)(v85 + 28) + 3 + v82 + (unsigned int)(v20 > 4u));
  v21 = *((_DWORD *)a6 + 2);
  v91 = 0;
  v90 = &v92;
  if ( v21 )
  {
    v22 = *a6;
    v23 = **a6;
    if ( v23 != -8 && v23 )
    {
      v26 = *a6;
    }
    else
    {
      v24 = v22 + 1;
      do
      {
        do
        {
          v25 = *v24;
          v26 = v24++;
        }
        while ( v25 == -8 );
      }
      while ( !v25 );
    }
    v27 = &v22[v21];
    if ( v26 != v27 )
    {
      v28 = &v92;
      v29 = 0;
      v30 = 0;
      while ( 1 )
      {
        v31 = (__int64 *)*v26;
        v32 = v30;
        v33 = *(_QWORD *)*v26;
        v34 = (__m128i *)((char *)v28 + 24 * v30);
        if ( v30 >= v29 )
        {
          v69 = v31[1];
          v31 = (__int64 *)(v32 + 1);
          v106.m128i_i64[0] = *v26 + 16;
          v106.m128i_i64[1] = v33;
          v70 = &v106;
          v107.m128i_i64[0] = v69;
          if ( v29 < v32 + 1 )
          {
            if ( v28 > &v106 || v34 <= &v106 )
            {
              sub_C8D5F0((__int64)&v90, &v92, v32 + 1, 0x18u, (__int64)v31, v33);
              v28 = v90;
              v32 = (unsigned int)v91;
            }
            else
            {
              v75 = (char *)((char *)&v106 - (char *)v28);
              sub_C8D5F0((__int64)&v90, &v92, v32 + 1, 0x18u, (__int64)v31, v33);
              v28 = v90;
              v32 = (unsigned int)v91;
              v70 = (__m128i *)&v75[(_QWORD)v90];
            }
          }
          v71 = (__m128i *)((char *)v28 + 24 * v32);
          *v71 = _mm_loadu_si128(v70);
          v28 = v90;
          v71[1].m128i_i64[0] = v70[1].m128i_i64[0];
          v30 = v91 + 1;
          LODWORD(v91) = v91 + 1;
        }
        else
        {
          if ( v34 )
          {
            v34->m128i_i64[0] = *v26 + 16;
            v28 = v90;
            v34->m128i_i64[1] = v33;
            v34[1].m128i_i64[0] = v31[1];
            v30 = v91;
          }
          LODWORD(v91) = ++v30;
        }
        v35 = v26[1];
        v36 = (__int64)(v26 + 1);
        if ( v35 != -8 && v35 )
        {
          ++v26;
        }
        else
        {
          v37 = v26 + 2;
          do
          {
            do
            {
              v36 = *v37;
              v26 = v37++;
            }
            while ( v36 == -8 );
          }
          while ( !v36 );
        }
        if ( v26 == v27 )
          break;
        v29 = HIDWORD(v91);
      }
      v41 = 24LL * v30;
      v42 = (__m128i *)((char *)v28 + v41);
      if ( v28 != (__m128i *)&v28->m128i_i8[v41] )
      {
        _BitScanReverse64(&v43, 0xAAAAAAAAAAAAAAABLL * (v41 >> 3));
        sub_3218FB0((__int64)v28, (__m128i *)((char *)v28 + v41), 2LL * (int)(63 - (v43 ^ 0x3F)), v36, (__int64)v31);
        if ( (unsigned __int64)v41 <= 0x180 )
        {
          sub_32185D0(v28, v42);
        }
        else
        {
          v44 = v28 + 24;
          sub_32185D0(v28, v28 + 24);
          if ( &v28[24] != v42 )
          {
            do
            {
              v45 = v44;
              v44 = (__m128i *)((char *)v44 + 24);
              sub_32184B0(v45);
            }
            while ( v44 != v42 );
          }
        }
        v89 = (__m128i *)((char *)v90 + 24 * (unsigned int)v91);
        if ( v90 != v89 )
        {
          v46 = (unsigned __int64)v90;
          while ( 1 )
          {
            v50 = *(_QWORD *)(a1 + 8);
            v51 = *(_QWORD *)(v50 + 224);
            v52 = *(void (**)())(*(_QWORD *)v51 + 120LL);
            v106.m128i_i64[0] = (__int64)"DIE offset";
            LOWORD(v108) = 259;
            if ( v52 != nullsub_98 )
            {
              ((void (__fastcall *)(__int64, __m128i *, __int64))v52)(v51, &v106, 1);
              v50 = *(_QWORD *)(a1 + 8);
            }
            sub_31F0F00(v50, *(unsigned int *)(*(_QWORD *)(v46 + 16) + 16LL));
            if ( !a2 )
              goto LABEL_41;
            if ( *(_WORD *)(*(_QWORD *)(v46 + 16) + 28LL) == 17 )
            {
LABEL_51:
              v55 = 16;
              v54 = 0;
              v56 = 1;
            }
            else
            {
              v83 = *(_QWORD *)(v46 + 16);
              sub_3215160((__int64)&v101, v83, 71);
              if ( (_DWORD)v101 )
                v53 = v102;
              else
                v53 = v83;
              sub_3215160((__int64)&v106, v53, 63);
              v54 = v106.m128i_i32[0] == 0;
              switch ( *(_WORD *)(v83 + 28) )
              {
                case 2:
                case 4:
                case 0x13:
                case 0x17:
                  v73 = *(unsigned __int16 *)(*(_QWORD *)(v85 + 80) + 16LL);
                  if ( v73 > 0x2B )
                    goto LABEL_80;
                  v74 = (1LL << v73) & 0xC0206000010LL;
                  v55 = v74 == 0 ? 144 : 16;
                  v54 = v74 == 0;
                  goto LABEL_79;
                case 0x16:
                case 0x21:
                case 0x24:
                case 0x43:
LABEL_80:
                  v55 = 144;
                  v54 = 1;
LABEL_79:
                  v56 = 1;
                  break;
                case 0x28:
                  v55 = 160;
                  v54 = 1;
                  v56 = 2;
                  break;
                case 0x2E:
                  v56 = 3;
                  v55 = (v54 << 7) | 0x30;
                  break;
                case 0x34:
                  v56 = 2;
                  v55 = (v54 << 7) | 0x20;
                  break;
                case 0x39:
                  goto LABEL_51;
                default:
                  v55 = 0;
                  v54 = 0;
                  v56 = 0;
                  break;
              }
            }
            v80 = v55;
            v86 = v56;
            v81 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL);
            v84 = *(void (**)())(*(_QWORD *)v81 + 120LL);
            v57 = sub_E0CB60(v54);
            v97 = 1;
            v101 = v57;
            v105 = 261;
            v102 = v58;
            v95[0] = ", ";
            v96 = 3;
            v59.m128i_i64[0] = (__int64)sub_E0CAC0(v86);
            v93 = v59;
            v60 = v96;
            v92.m128i_i64[0] = (__int64)"Attributes: ";
            v61 = v80;
            LOWORD(v94) = 1283;
            if ( !v96 )
            {
              LOWORD(v100) = 256;
LABEL_64:
              LOWORD(v108) = 256;
LABEL_65:
              if ( v84 == nullsub_98 )
                goto LABEL_66;
              goto LABEL_62;
            }
            if ( v96 != 1 )
              break;
            v67 = _mm_loadu_si128(&v92);
            v68 = _mm_loadu_si128(&v93);
            v100 = v94;
            v63 = v105;
            v98 = v67;
            v99 = v68;
            if ( !(_BYTE)v105 )
              goto LABEL_64;
            if ( (_BYTE)v105 == 1 )
            {
LABEL_76:
              v72 = _mm_loadu_si128(&v99);
              v106 = _mm_loadu_si128(&v98);
              v108 = v100;
              v107 = v72;
              goto LABEL_65;
            }
            if ( BYTE1(v100) != 1 )
              goto LABEL_58;
            v65 = 3;
            v78 = v98.m128i_i64[1];
            v64 = (__m128i *)v98.m128i_i64[0];
LABEL_59:
            if ( HIBYTE(v105) == 1 )
            {
              v77 = v102;
              v66 = (const char **)v101;
            }
            else
            {
              v66 = &v101;
              v63 = 2;
            }
            v106.m128i_i64[0] = (__int64)v64;
            v107.m128i_i64[0] = (__int64)v66;
            v106.m128i_i64[1] = v78;
            LOBYTE(v108) = v65;
            v107.m128i_i64[1] = v77;
            BYTE1(v108) = v63;
            if ( v84 == nullsub_98 )
              goto LABEL_66;
LABEL_62:
            ((void (__fastcall *)(__int64, __m128i *, __int64))v84)(v81, &v106, 1);
            v61 = v80;
LABEL_66:
            sub_31DC9D0(*(_QWORD *)(a1 + 8), v61);
LABEL_41:
            v47 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL);
            v48 = *(void (**)())(*(_QWORD *)v47 + 120LL);
            v106.m128i_i64[0] = (__int64)"External Name";
            LOWORD(v108) = 259;
            if ( v48 != nullsub_98 )
            {
              ((void (__fastcall *)(__int64, __m128i *, __int64))v48)(v47, &v106, 1);
              v47 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL);
            }
            v49 = *(_QWORD *)(v46 + 8);
            v46 += 24LL;
            (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v47 + 512LL))(
              v47,
              *(_QWORD *)(v46 - 24),
              v49 + 1);
            if ( v89 == (__m128i *)v46 )
              goto LABEL_30;
          }
          if ( v97 == 1 )
          {
            v62 = (_QWORD *)v95[0];
            v79 = v95[1];
          }
          else
          {
            v62 = v95;
            v60 = 2;
          }
          BYTE1(v100) = v60;
          v63 = v105;
          v99.m128i_i64[0] = (__int64)v62;
          v98.m128i_i64[0] = (__int64)&v92;
          LOBYTE(v100) = 2;
          v99.m128i_i64[1] = v79;
          if ( !(_BYTE)v105 )
            goto LABEL_64;
          if ( (_BYTE)v105 == 1 )
            goto LABEL_76;
LABEL_58:
          v64 = &v98;
          v65 = 2;
          goto LABEL_59;
        }
      }
    }
  }
LABEL_30:
  v38 = *(_QWORD *)(a1 + 8);
  v39 = *(_QWORD *)(v38 + 224);
  v40 = *(void (**)())(*(_QWORD *)v39 + 120LL);
  v106.m128i_i64[0] = (__int64)"End Mark";
  LOWORD(v108) = 259;
  if ( v40 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, __m128i *, __int64))v40)(v39, &v106, 1);
    v38 = *(_QWORD *)(a1 + 8);
  }
  sub_31F0F00(v38, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 208LL))(
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
    v76,
    0);
  if ( v90 != &v92 )
    _libc_free((unsigned __int64)v90);
}
