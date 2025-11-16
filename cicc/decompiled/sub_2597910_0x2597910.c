// Function: sub_2597910
// Address: 0x2597910
//
__int64 __fastcall sub_2597910(__int64 *a1, __int64 a2, unsigned __int8 a3)
{
  __int64 *v3; // r14
  __int64 v4; // r12
  _BYTE *v5; // rbx
  __int64 v6; // r13
  unsigned __int64 v7; // rdi
  __m128i v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  int v13; // ebx
  __int64 v14; // rax
  unsigned __int8 *v15; // rax
  unsigned __int8 **v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  unsigned __int8 *v20; // rsi
  unsigned __int8 **v21; // rax
  unsigned int v22; // r12d
  __int64 v24; // rcx
  unsigned __int8 *v25; // r12
  unsigned __int8 v26; // al
  char v27; // al
  unsigned __int8 v28; // al
  char v29; // al
  __int64 v30; // r12
  unsigned __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // r12
  __int64 (__fastcall *v35)(__int64); // rax
  __int64 v36; // rdi
  _BYTE *v37; // rax
  __int64 v38; // r15
  __int64 v39; // r13
  unsigned __int64 v40; // rax
  __int64 v41; // rdx
  _QWORD *v42; // rax
  __int64 v43; // r9
  __int64 (__fastcall *v44)(__int64, __int64, __int64, __int64, __int64, __int64); // rsi
  _QWORD *v45; // r12
  __int64 v46; // rax
  __int64 v47; // r8
  _QWORD *v48; // rcx
  __int64 v49; // rdx
  __int64 v50; // rcx
  unsigned __int64 *v51; // r13
  __int64 *v52; // rax
  _QWORD *v53; // r14
  __int64 v54; // rdi
  unsigned __int64 *v55; // r12
  __int64 *v56; // r15
  unsigned __int64 v57; // rax
  unsigned int v58; // eax
  __int64 (__fastcall *v59)(__int64); // rax
  __int64 v60; // rdi
  __int64 v61; // r13
  __int64 v62; // rdi
  __int64 v63; // r15
  _BYTE *v64; // rbx
  unsigned __int64 v65; // rdi
  unsigned __int64 v66; // rax
  __int64 v67; // rdx
  __int64 v68; // r15
  __int64 (__fastcall *v69)(__int64); // rax
  __int64 v70; // rdx
  __int64 v71; // rcx
  __int64 v72; // r8
  __int64 v73; // r9
  char v74; // al
  __int64 (__fastcall *v75)(__int64, __int64, __int64, __int64, __int64, __int64); // [rsp-8h] [rbp-168h]
  int v76; // [rsp+4h] [rbp-15Ch]
  __int64 v77; // [rsp+20h] [rbp-140h]
  _BYTE *v78; // [rsp+38h] [rbp-128h]
  __int64 v79; // [rsp+38h] [rbp-128h]
  unsigned int v81; // [rsp+44h] [rbp-11Ch]
  unsigned __int8 v83; // [rsp+56h] [rbp-10Ah] BYREF
  char v84; // [rsp+57h] [rbp-109h] BYREF
  unsigned __int8 *v85; // [rsp+58h] [rbp-108h] BYREF
  unsigned __int8 *v86; // [rsp+60h] [rbp-100h] BYREF
  unsigned __int64 v87; // [rsp+68h] [rbp-F8h] BYREF
  __m128i v88; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v89; // [rsp+80h] [rbp-E0h]
  _BYTE **v90; // [rsp+88h] [rbp-D8h]
  _BYTE *v91; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v92; // [rsp+98h] [rbp-C8h]
  _BYTE v93[48]; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v94; // [rsp+D0h] [rbp-90h] BYREF
  unsigned __int8 **v95; // [rsp+D8h] [rbp-88h]
  __int64 v96; // [rsp+E0h] [rbp-80h]
  int v97; // [rsp+E8h] [rbp-78h]
  char v98; // [rsp+ECh] [rbp-74h]
  char v99; // [rsp+F0h] [rbp-70h] BYREF

  v3 = a1;
  v4 = *a1;
  v5 = (_BYTE *)a1[3];
  v95 = (unsigned __int8 **)&v99;
  v6 = a1[2];
  v7 = a1[1];
  v91 = v93;
  v92 = 0x300000000LL;
  v94 = 0;
  v96 = 8;
  v97 = 0;
  v98 = 1;
  v81 = a3;
  v8.m128i_i64[0] = sub_250D2C0(v7, 0);
  v88 = v8;
  if ( (unsigned __int8)sub_2526B50(v4, &v88, v6, (__int64)&v91, v81, v5, 1u) )
  {
    v83 = 0;
    if ( (_DWORD)v92 )
    {
      v13 = 0;
      v14 = 0;
      while ( 1 )
      {
        v85 = *(unsigned __int8 **)&v91[16 * v14];
        v15 = sub_98ACB0(v85, 6u);
        v20 = v15;
        if ( !v15 )
          v20 = v85;
        v86 = v15;
        if ( !v98 )
          goto LABEL_18;
        v21 = v95;
        v17 = HIDWORD(v96);
        v16 = &v95[HIDWORD(v96)];
        if ( v95 != v16 )
        {
          while ( v20 != *v21 )
          {
            if ( v16 == ++v21 )
              goto LABEL_25;
          }
          goto LABEL_11;
        }
LABEL_25:
        if ( HIDWORD(v96) < (unsigned int)v96 )
        {
          v24 = (unsigned int)++HIDWORD(v96);
          *v16 = v20;
          ++v94;
        }
        else
        {
LABEL_18:
          sub_C8CC70((__int64)&v94, (__int64)v20, (__int64)v16, v17, v18, v19);
          if ( !(_BYTE)v16 )
            goto LABEL_11;
        }
        v25 = v85;
        if ( !v86 || v86 == v85 )
        {
          v28 = *v85;
          if ( *v85 <= 0x1Cu )
          {
LABEL_30:
            v29 = sub_25784B0(a2, (__int64 *)&v85, (__int64)v16, v24, v18, v19);
            v83 |= v29;
            v14 = (unsigned int)(v13 + 1);
            v13 = v14;
            if ( (unsigned int)v92 <= (unsigned int)v14 )
              goto LABEL_12;
          }
          else if ( v28 == 86 )
          {
            v37 = (_BYTE *)v3[3];
            v38 = v3[2];
            v39 = *v3;
            v84 = 0;
            v78 = v37;
            v40 = sub_250D2C0((unsigned __int64)v85, 0);
            v42 = (_QWORD *)sub_252AE70(v39, v40, v41, v38, 1, 0, 1);
            v44 = v75;
            v45 = v42;
            v88.m128i_i64[0] = (__int64)&v84;
            v88.m128i_i64[1] = a2;
            if ( !v42 )
              goto LABEL_72;
            v46 = *v42;
            v47 = *(_QWORD *)(*v45 + 112LL);
            if ( (__int64 (__fastcall *)(__int64, __int64 (__fastcall *)(__int64, unsigned __int64), __int64, char))v47 == sub_254E4A0 )
            {
              if ( *((_BYTE *)v45 + 97) )
              {
                v48 = v45 + 27;
                if ( a3 == 1 )
                  v48 = v45 + 13;
                v49 = v48[4];
                v50 = *((unsigned int *)v48 + 10);
                v51 = (unsigned __int64 *)(v49 + 8 * v50);
                if ( (unsigned __int64 *)v49 != v51 )
                {
                  v52 = v3;
                  v53 = v45;
                  v54 = a2;
                  v55 = (unsigned __int64 *)v49;
                  v56 = v52;
                  while ( 1 )
                  {
                    v57 = *v55;
                    v44 = (__int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64, __int64))&v87;
                    ++v55;
                    v87 = v57;
                    v58 = sub_25784B0(v54, (__int64 *)&v87, v49, v50, v47, v43);
                    v47 = v58;
                    *(_BYTE *)v88.m128i_i64[0] |= v58;
                    if ( v51 == v55 )
                      break;
                    v54 = v88.m128i_i64[1];
                  }
                  v45 = v53;
                  v3 = v56;
                  v46 = *v45;
                }
              }
              else
              {
                v44 = (__int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64, __int64))&v87;
                v87 = sub_250D070(v45 + 9);
                v74 = sub_25784B0(v88.m128i_i64[1], (__int64 *)&v87, v70, v71, v72, v73);
                *(_BYTE *)v88.m128i_i64[0] |= v74;
                v46 = *v45;
              }
            }
            else
            {
              v44 = sub_25789A0;
              if ( !((unsigned __int8 (__fastcall *)(_QWORD *, __int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64, __int64), __m128i *, _QWORD))v47)(
                      v45,
                      sub_25789A0,
                      &v88,
                      v81) )
LABEL_72:
                BUG();
              v46 = *v45;
            }
            v59 = *(__int64 (__fastcall **)(__int64))(v46 + 48);
            if ( v59 == sub_25352C0 )
              v60 = (__int64)(v45 + 11);
            else
              v60 = ((__int64 (__fastcall *)(_QWORD *, __int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64, __int64)))v59)(
                      v45,
                      v44);
            *v78 |= (*(unsigned __int8 (__fastcall **)(__int64, __int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64, __int64)))(*(_QWORD *)v60 + 24LL))(
                      v60,
                      v44)
                  ^ 1;
            v83 |= v84;
            v14 = (unsigned int)(v13 + 1);
            v13 = v14;
            if ( (unsigned int)v92 <= (unsigned int)v14 )
            {
LABEL_12:
              v22 = v83;
              goto LABEL_13;
            }
          }
          else
          {
            if ( v28 != 84 )
              goto LABEL_30;
            if ( (*((_DWORD *)v85 + 1) & 0x7FFFFFF) != 0 )
            {
              v76 = v13;
              v61 = 0;
              v77 = 32LL * (*((_DWORD *)v85 + 1) & 0x7FFFFFF);
              do
              {
                v63 = *v3;
                v64 = (_BYTE *)v3[3];
                v65 = *(_QWORD *)(*((_QWORD *)v25 - 1) + v61);
                v79 = v3[2];
                LOBYTE(v87) = 0;
                v66 = sub_250D2C0(v65, 0);
                v68 = sub_252AE70(v63, v66, v67, v79, 1, 0, 1);
                v88.m128i_i64[0] = (__int64)&v87;
                v88.m128i_i64[1] = a2;
                if ( !v68
                  || !(*(unsigned __int8 (__fastcall **)(__int64, __int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64, __int64), __m128i *, _QWORD))(*(_QWORD *)v68 + 112LL))(
                        v68,
                        sub_25789A0,
                        &v88,
                        v81) )
                {
                  BUG();
                }
                v69 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v68 + 48LL);
                if ( v69 == sub_25352C0 )
                  v62 = v68 + 88;
                else
                  v62 = v69(v68);
                v61 += 32;
                *v64 |= (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v62 + 24LL))(v62) ^ 1;
                v83 |= v87;
              }
              while ( v61 != v77 );
              v14 = (unsigned int)(v76 + 1);
              v13 = v14;
              if ( (unsigned int)v92 <= (unsigned int)v14 )
                goto LABEL_12;
            }
            else
            {
LABEL_11:
              v14 = (unsigned int)(v13 + 1);
              v13 = v14;
              if ( (unsigned int)v92 <= (unsigned int)v14 )
                goto LABEL_12;
            }
          }
        }
        else
        {
          v26 = *v86;
          if ( *v86 > 0x1Cu )
          {
            if ( v26 == 60 )
              goto LABEL_23;
LABEL_33:
            v30 = *v3;
            v31 = sub_250D2C0((unsigned __int64)v86, 0);
            v33 = sub_252AE70(v30, v31, v32, v3[2], 1, 0, 1);
            v34 = v33;
            v88.m128i_i64[0] = (__int64)&v86;
            v88.m128i_i64[1] = (__int64)&v83;
            v89 = a2;
            v90 = &v91;
            if ( !v33
              || !(*(unsigned __int8 (__fastcall **)(__int64, __int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64, __int64), __m128i *, _QWORD))(*(_QWORD *)v33 + 112LL))(
                    v33,
                    sub_2578910,
                    &v88,
                    v81) )
            {
              BUG();
            }
            v35 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v34 + 48LL);
            if ( v35 == sub_25352C0 )
              v36 = v34 + 88;
            else
              v36 = v35(v34);
            *(_BYTE *)v3[3] |= (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v36 + 24LL))(v36) ^ 1;
            v14 = (unsigned int)(v13 + 1);
            v13 = v14;
            if ( (unsigned int)v92 <= (unsigned int)v14 )
              goto LABEL_12;
          }
          else
          {
            if ( v26 > 3u )
              goto LABEL_33;
LABEL_23:
            v27 = sub_25784B0(a2, (__int64 *)&v86, (__int64)v16, v24, v18, v19);
            v83 |= v27;
            v14 = (unsigned int)(v13 + 1);
            v13 = v14;
            if ( (unsigned int)v92 <= (unsigned int)v14 )
              goto LABEL_12;
          }
        }
      }
    }
    v22 = 0;
  }
  else
  {
    v88.m128i_i64[0] = v3[1];
    v22 = sub_25784B0(a2, v88.m128i_i64, v9, v10, v11, v12);
  }
LABEL_13:
  if ( v91 != v93 )
    _libc_free((unsigned __int64)v91);
  if ( !v98 )
    _libc_free((unsigned __int64)v95);
  return v22;
}
