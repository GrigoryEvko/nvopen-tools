// Function: sub_25BB3B0
// Address: 0x25bb3b0
//
__int64 __fastcall sub_25BB3B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r15
  __int64 v5; // r14
  __int64 v6; // r13
  _BYTE *v7; // rax
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rbx
  void *v10; // r12
  char *v11; // rcx
  size_t v12; // rbx
  const char *v13; // rax
  __int64 v14; // rdx
  int v15; // eax
  int v16; // r12d
  __int64 v17; // r12
  void *v18; // rbx
  __int64 v19; // r15
  __int64 v20; // r14
  _BYTE *v21; // rax
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rbx
  void *v24; // r12
  char *v25; // rcx
  size_t v26; // rbx
  const char *v27; // rax
  __int64 v28; // rdx
  int v29; // eax
  int v30; // r12d
  __int64 v31; // r12
  void *v32; // rbx
  unsigned __int64 v34; // rax
  unsigned __int64 v35; // rdx
  unsigned __int64 v36; // rcx
  _BYTE *v37; // rax
  __int64 v38; // r15
  size_t v39; // rax
  size_t v40; // rdx
  size_t v41; // rcx
  size_t v42; // r8
  char *v43; // rcx
  int v44; // eax
  char v45; // al
  _QWORD *v46; // rax
  void *v47; // rdx
  __int64 v48; // r8
  __m128i *v49; // rdi
  size_t v50; // rax
  __m128i si128; // xmm0
  _QWORD *v52; // rax
  __m128i *v53; // rdx
  __int64 v54; // rdi
  __m128i v55; // xmm0
  __int64 v56; // rax
  __m128i *v57; // rdx
  __m128i v58; // xmm0
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rax
  __m128i *v62; // rdx
  __int64 v63; // [rsp+8h] [rbp-128h]
  __int64 v64; // [rsp+10h] [rbp-120h]
  __int64 v66; // [rsp+20h] [rbp-110h]
  __int64 v67; // [rsp+20h] [rbp-110h]
  __int64 v68; // [rsp+28h] [rbp-108h]
  int v69; // [rsp+28h] [rbp-108h]
  size_t v70; // [rsp+28h] [rbp-108h]
  __int64 v71; // [rsp+30h] [rbp-100h]
  __int64 v72; // [rsp+30h] [rbp-100h]
  char v73; // [rsp+30h] [rbp-100h]
  __int64 v74; // [rsp+38h] [rbp-F8h]
  __int64 v75; // [rsp+38h] [rbp-F8h]
  char v76; // [rsp+48h] [rbp-E8h]
  char v77; // [rsp+57h] [rbp-D9h] BYREF
  __int64 v78; // [rsp+58h] [rbp-D8h] BYREF
  __int64 *v79; // [rsp+60h] [rbp-D0h] BYREF
  char v80; // [rsp+70h] [rbp-C0h]
  __int64 v81; // [rsp+80h] [rbp-B0h]
  unsigned __int64 v82; // [rsp+88h] [rbp-A8h]
  void *src; // [rsp+90h] [rbp-A0h] BYREF
  size_t n; // [rsp+98h] [rbp-98h]
  _QWORD v85[4]; // [rsp+A0h] [rbp-90h] BYREF
  void *s1; // [rsp+C0h] [rbp-70h] BYREF
  unsigned __int64 v87; // [rsp+C8h] [rbp-68h]
  __int16 v88; // [rsp+E0h] [rbp-50h]
  unsigned int v89; // [rsp+ECh] [rbp-44h]
  __int64 v90; // [rsp+F0h] [rbp-40h] BYREF
  unsigned __int64 v91; // [rsp+F8h] [rbp-38h]

  v76 = 0;
  if ( qword_4FF00D0 )
  {
    v88 = 260;
    s1 = &qword_4FF00C8;
    sub_C7EAD0((__int64)&v79, (const char ***)&s1, 0, 1u, 0);
    v76 = v80 & 1;
    if ( (v80 & 1) != 0 )
      sub_C64ED0("Cannot open CSV file.", 1u);
    sub_C7DA90(&v78, v79[1], v79[2] - v79[1], byte_3F871B3, 0, 1);
    sub_C7C840((__int64)&s1, v78, 1, 0);
    v73 = v88;
    while ( (_BYTE)v88 )
    {
      LOBYTE(v85[0]) = 44;
      v34 = sub_C931B0(&v90, v85, 1u, 0);
      if ( v34 != -1 )
      {
        v35 = v91;
        v36 = v34 + 1;
        if ( v34 + 1 <= v91 )
        {
          v81 = v90;
          if ( v34 <= v91 )
            v35 = v34;
          n = v91 - v36;
          src = (void *)(v90 + v36);
          v82 = v35;
          if ( v91 != v36 )
          {
            v37 = sub_BA8CB0(a3, v90, v35);
            v38 = (__int64)v37;
            if ( v37 )
            {
              if ( !sub_B2FC80((__int64)v37) )
              {
                v77 = 61;
                v39 = sub_C931B0((__int64 *)&src, &v77, 1u, 0);
                if ( v39 == -1 )
                  goto LABEL_73;
                v40 = n;
                v41 = v39 + 1;
                if ( v39 + 1 > n )
                  goto LABEL_73;
                v85[0] = src;
                v42 = n - v41;
                if ( v39 <= n )
                  v40 = v39;
                v43 = (char *)src + v41;
                v85[3] = v42;
                v85[2] = v43;
                v85[1] = v40;
                if ( !v42 )
                {
LABEL_73:
                  v44 = sub_A6E860((__int64)src, n);
                  if ( v44 && (v69 = v44, (v45 = sub_A719F0(v44)) != 0) )
                  {
                    v76 = v45;
                    sub_B2CD30(v38, v69);
                  }
                  else
                  {
                    v46 = sub_CB72A0();
                    v47 = (void *)v46[4];
                    v48 = (__int64)v46;
                    if ( v46[3] - (_QWORD)v47 <= 0xAu )
                    {
                      v59 = sub_CB6200((__int64)v46, "Cannot add ", 0xBu);
                      v49 = *(__m128i **)(v59 + 32);
                      v48 = v59;
                    }
                    else
                    {
                      qmemcpy(v47, "Cannot add ", 11);
                      v49 = (__m128i *)(v46[4] + 11LL);
                      v46[4] = v49;
                    }
                    v50 = *(_QWORD *)(v48 + 24) - (_QWORD)v49;
                    if ( v50 < n )
                    {
                      v60 = sub_CB6200(v48, (unsigned __int8 *)src, n);
                      v49 = *(__m128i **)(v60 + 32);
                      v48 = v60;
                      v50 = *(_QWORD *)(v60 + 24) - (_QWORD)v49;
                    }
                    else if ( n )
                    {
                      v67 = v48;
                      v70 = n;
                      memcpy(v49, src, n);
                      v48 = v67;
                      v61 = *(_QWORD *)(v67 + 24);
                      v62 = (__m128i *)(*(_QWORD *)(v67 + 32) + v70);
                      *(_QWORD *)(v67 + 32) = v62;
                      v49 = v62;
                      v50 = v61 - (_QWORD)v62;
                    }
                    if ( v50 <= 0x16 )
                    {
                      sub_CB6200(v48, " as an attribute name.\n", 0x17u);
                    }
                    else
                    {
                      si128 = _mm_load_si128((const __m128i *)&xmmword_438ADF0);
                      v49[1].m128i_i32[0] = 1835101728;
                      v49[1].m128i_i16[2] = 11877;
                      v49[1].m128i_i8[6] = 10;
                      *v49 = si128;
                      *(_QWORD *)(v48 + 32) += 23LL;
                    }
                  }
                }
                else
                {
                  sub_B2CD60(v38, src, v40, v43, v42);
                  v76 = v73;
                }
              }
            }
            else
            {
              v52 = sub_CB72A0();
              v53 = (__m128i *)v52[4];
              v54 = (__int64)v52;
              if ( v52[3] - (_QWORD)v53 <= 0x1Cu )
              {
                v54 = sub_CB6200((__int64)v52, "Function in CSV file at line ", 0x1Du);
              }
              else
              {
                v55 = _mm_load_si128((const __m128i *)&xmmword_438AE00);
                qmemcpy(&v53[1], "file at line ", 13);
                *v53 = v55;
                v52[4] += 29LL;
              }
              v56 = sub_CB59F0(v54, v89);
              v57 = *(__m128i **)(v56 + 32);
              if ( *(_QWORD *)(v56 + 24) - (_QWORD)v57 <= 0x10u )
              {
                sub_CB6200(v56, " does not exist.\n", 0x11u);
              }
              else
              {
                v58 = _mm_load_si128((const __m128i *)&xmmword_438AE10);
                v57[1].m128i_i8[0] = 10;
                *v57 = v58;
                *(_QWORD *)(v56 + 32) += 17LL;
              }
            }
          }
        }
      }
      sub_C7C5C0((__int64)&s1);
    }
    if ( v78 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v78 + 8LL))(v78);
    if ( (v80 & 1) == 0 && v79 )
      (*(void (__fastcall **)(__int64 *))(*v79 + 8))(v79);
  }
  v4 = qword_4FF02D0;
  v5 = qword_4FF02C8;
  v63 = a1 + 32;
  v64 = a1 + 80;
  if ( qword_4FF02D0 != qword_4FF02C8 || qword_4FF01D0 != qword_4FF01C8 )
  {
    v66 = a3 + 24;
    v68 = *(_QWORD *)(a3 + 32);
    if ( a3 + 24 == v68 )
      goto LABEL_50;
    while ( 1 )
    {
      v6 = v68 - 56;
      if ( !v68 )
        v6 = 0;
      if ( v4 != v5 )
        break;
LABEL_24:
      v19 = qword_4FF01D0;
      v20 = qword_4FF01C8;
      if ( qword_4FF01C8 != qword_4FF01D0 )
      {
        while ( 1 )
        {
          v31 = *(_QWORD *)(v20 + 8);
          v32 = *(void **)v20;
          if ( v31 && (v21 = memchr(*(const void **)v20, 58, *(_QWORD *)(v20 + 8))) != 0 && v21 - (_BYTE *)v32 != -1 )
          {
            s1 = v32;
            v87 = v31;
            LOBYTE(v85[0]) = 58;
            v22 = sub_C931B0((__int64 *)&s1, v85, 1u, 0);
            if ( v22 == -1 )
            {
              v72 = 0;
              v24 = s1;
              v75 = 0;
              v26 = v87;
            }
            else
            {
              v23 = v22 + 1;
              v24 = s1;
              if ( v22 + 1 > v87 )
              {
                v72 = 0;
                v23 = v87;
              }
              else
              {
                v72 = v87 - v23;
              }
              v25 = (char *)s1 + v23;
              v26 = v87;
              v75 = (__int64)v25;
              if ( v22 <= v87 )
                v26 = v22;
            }
            v27 = sub_BD5D20(v6);
            if ( v26 != v28 || v26 && memcmp(v24, v27, v26) )
              goto LABEL_37;
            v29 = sub_A6E860(v75, v72);
            v30 = v29;
            if ( !v29 )
              goto LABEL_37;
LABEL_40:
            sub_A719F0(v29);
            if ( !(unsigned __int8)sub_B2D610(v6, v30) )
              goto LABEL_37;
            v20 += 32;
            sub_B2D470(v6, v30);
            if ( v19 == v20 )
              break;
          }
          else
          {
            v29 = sub_A6E860((__int64)v32, v31);
            v30 = v29;
            if ( v29 )
              goto LABEL_40;
LABEL_37:
            v20 += 32;
            if ( v19 == v20 )
              break;
          }
        }
      }
      v68 = *(_QWORD *)(v68 + 8);
      if ( v66 == v68 )
        goto LABEL_50;
      v5 = qword_4FF02C8;
      v4 = qword_4FF02D0;
    }
    while ( 1 )
    {
      v17 = *(_QWORD *)(v5 + 8);
      v18 = *(void **)v5;
      if ( v17 && (v7 = memchr(*(const void **)v5, 58, *(_QWORD *)(v5 + 8))) != 0 && v7 - (_BYTE *)v18 != -1 )
      {
        s1 = v18;
        v87 = v17;
        LOBYTE(v85[0]) = 58;
        v8 = sub_C931B0((__int64 *)&s1, v85, 1u, 0);
        if ( v8 == -1 )
        {
          v71 = 0;
          v10 = s1;
          v74 = 0;
          v12 = v87;
        }
        else
        {
          v9 = v8 + 1;
          v10 = s1;
          if ( v8 + 1 > v87 )
          {
            v71 = 0;
            v9 = v87;
          }
          else
          {
            v71 = v87 - v9;
          }
          v11 = (char *)s1 + v9;
          v12 = v87;
          v74 = (__int64)v11;
          if ( v8 <= v87 )
            v12 = v8;
        }
        v13 = sub_BD5D20(v6);
        if ( v12 != v14 || v12 && memcmp(v10, v13, v12) )
          goto LABEL_19;
        v15 = sub_A6E860(v74, v71);
        v16 = v15;
        if ( !v15 )
          goto LABEL_19;
LABEL_22:
        sub_A719F0(v15);
        if ( (unsigned __int8)sub_B2D610(v6, v16) )
          goto LABEL_19;
        v5 += 32;
        sub_B2CD30(v6, v16);
        if ( v4 == v5 )
          goto LABEL_24;
      }
      else
      {
        v15 = sub_A6E860((__int64)v18, v17);
        v16 = v15;
        if ( v15 )
          goto LABEL_22;
LABEL_19:
        v5 += 32;
        if ( v4 == v5 )
          goto LABEL_24;
      }
    }
  }
  if ( !v76 )
  {
    *(_QWORD *)(a1 + 8) = v63;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v64;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)a1 = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    return a1;
  }
LABEL_50:
  memset((void *)a1, 0, 0x60u);
  *(_QWORD *)(a1 + 8) = v63;
  *(_DWORD *)(a1 + 16) = 2;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 56) = v64;
  *(_DWORD *)(a1 + 64) = 2;
  *(_BYTE *)(a1 + 76) = 1;
  return a1;
}
