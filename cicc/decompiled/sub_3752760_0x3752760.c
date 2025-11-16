// Function: sub_3752760
// Address: 0x3752760
//
__int64 __fastcall sub_3752760(
        __int64 *a1,
        __int64 *a2,
        unsigned __int64 a3,
        unsigned int a4,
        unsigned int a5,
        __int64 a6,
        __int64 a7,
        char a8,
        unsigned __int8 a9,
        unsigned __int8 a10)
{
  int v14; // eax
  bool v15; // r10
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rsi
  __int64 v20; // rax
  __int64 *v21; // rdx
  unsigned int v22; // eax
  __int64 v23; // rcx
  __int64 v24; // rdi
  __int64 v25; // rsi
  unsigned int v26; // r11d
  __int64 v27; // rbx
  _QWORD *v28; // rax
  _QWORD *v29; // rax
  __int64 v30; // rbx
  __int64 v31; // r8
  __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // rdi
  __int64 (*v35)(); // rax
  __int64 v36; // rax
  __int64 v37; // rdi
  __int64 v38; // rsi
  __int64 v39; // rcx
  int v40; // eax
  __int64 v41; // rdx
  __int64 v42; // rdi
  __int64 v43; // rsi
  __int64 v44; // rdi
  __int64 v45; // rsi
  __int64 v46; // rax
  __int32 v47; // eax
  unsigned __int8 *v48; // rsi
  bool v49; // r10
  __int32 v50; // ebx
  __int64 v51; // rax
  __int64 v52; // r9
  __int64 v53; // rcx
  __int64 *v54; // rsi
  __int64 v55; // rdi
  _QWORD *v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rdi
  __int64 v59; // rsi
  __int64 v60; // rax
  int v61; // eax
  __int64 v62; // rdi
  __int64 v63; // rsi
  unsigned __int8 v64; // al
  int v65; // edx
  unsigned int v66; // edx
  __int64 v67; // rsi
  __int64 v68; // rdi
  int v69; // ebx
  int v70; // edx
  __int64 v71; // rdx
  int v72; // ecx
  __int64 v73; // rcx
  __int64 v74; // [rsp+10h] [rbp-C0h]
  __int64 v75; // [rsp+18h] [rbp-B8h]
  __int64 v76; // [rsp+18h] [rbp-B8h]
  __int64 v77; // [rsp+20h] [rbp-B0h]
  unsigned int v78; // [rsp+28h] [rbp-A8h]
  __int64 v79; // [rsp+28h] [rbp-A8h]
  __int64 v80; // [rsp+28h] [rbp-A8h]
  bool v81; // [rsp+28h] [rbp-A8h]
  __int64 v82; // [rsp+28h] [rbp-A8h]
  bool v83; // [rsp+28h] [rbp-A8h]
  __int64 v84; // [rsp+30h] [rbp-A0h]
  __int64 (__fastcall *v85)(__int64, unsigned __int16); // [rsp+30h] [rbp-A0h]
  bool v86; // [rsp+30h] [rbp-A0h]
  __int64 v87; // [rsp+30h] [rbp-A0h]
  bool v88; // [rsp+30h] [rbp-A0h]
  __int64 v89; // [rsp+30h] [rbp-A0h]
  int v90; // [rsp+38h] [rbp-98h]
  bool v91; // [rsp+38h] [rbp-98h]
  unsigned __int8 *v92; // [rsp+48h] [rbp-88h] BYREF
  __int64 v93[4]; // [rsp+50h] [rbp-80h] BYREF
  __m128i v94; // [rsp+70h] [rbp-60h] BYREF
  __int64 v95; // [rsp+80h] [rbp-50h]
  __int64 v96; // [rsp+88h] [rbp-48h]
  __int64 v97; // [rsp+90h] [rbp-40h]

  v14 = *(_DWORD *)(a3 + 24);
  if ( v14 < 0 )
    return sub_3752220(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
  if ( v14 == 11 || v14 == 35 )
  {
    v20 = *(_QWORD *)(a3 + 96);
    v21 = *(__int64 **)(v20 + 24);
    v22 = *(_DWORD *)(v20 + 32);
    if ( v22 > 0x40 )
    {
      v23 = *v21;
    }
    else
    {
      v23 = 0;
      if ( v22 )
        v23 = (__int64)((_QWORD)v21 << (64 - (unsigned __int8)v22)) >> (64 - (unsigned __int8)v22);
    }
    v24 = a2[1];
    v25 = *a2;
    v96 = v23;
    v94.m128i_i64[0] = 1;
    v95 = 0;
    return sub_2E8EAD0(v24, v25, &v94);
  }
  else
  {
    v15 = v14 == 36 || v14 == 12;
    if ( v15 )
    {
      v94.m128i_i64[0] = 3;
      v16 = *(_QWORD *)(a3 + 96);
      v17 = a2[1];
LABEL_6:
      v18 = *a2;
      v95 = 0;
      v96 = v16;
      return sub_2E8EAD0(v17, v18, &v94);
    }
    if ( v14 == 9 )
    {
      v90 = *(_DWORD *)(a3 + 96);
      v26 = *(unsigned __int16 *)(*(_QWORD *)(a3 + 48) + 16LL * a4);
      if ( a6 )
      {
        v27 = a1[3];
        v78 = *(unsigned __int16 *)(*(_QWORD *)(a3 + 48) + 16LL * a4);
        v84 = a6;
        v28 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64, __int64))(*(_QWORD *)a1[2] + 16LL))(
                          a1[2],
                          a6,
                          a5,
                          v27,
                          *a1);
        v29 = sub_2FF6410(v27, v28);
        v26 = v78;
        a6 = v84;
        v15 = 0;
        v30 = (__int64)v29;
        if ( !(_WORD)v78
          || (v31 = a1[4], v32 = (unsigned __int16)v78, !*(_QWORD *)(v31 + 8LL * (unsigned __int16)v78 + 112)) )
        {
LABEL_30:
          if ( a5 >= *(unsigned __int16 *)(a6 + 2) )
            v15 = ((*(_QWORD *)(a6 + 24) >> 1) ^ 1) & 1;
LABEL_32:
          v37 = a2[1];
          v94.m128i_i64[0] = 0;
          v38 = *a2;
          *(__int32 *)((char *)v94.m128i_i32 + 3) = (unsigned __int8)(32 * v15);
          v95 = 0;
          *(__int32 *)((char *)v94.m128i_i32 + 2) = v94.m128i_i16[1] & 0xF00F;
          v94.m128i_i32[0] &= 0xFFF000FF;
          v94.m128i_i32[2] = v90;
          v96 = 0;
          v97 = 0;
          return sub_2E8EAD0(v37, v38, &v94);
        }
        v33 = 1;
        v85 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v31 + 552LL);
        if ( (*(_BYTE *)(a3 + 32) & 4) != 0 )
        {
LABEL_21:
          if ( v85 == sub_2EC09E0 )
          {
            v36 = *(_QWORD *)(v31 + 8 * v32 + 112);
          }
          else
          {
            v77 = a6;
            v83 = v15;
            v36 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64))v85)(v31, v26, v33);
            a6 = v77;
            v15 = v83;
          }
          if ( v30 != 0 && v36 != 0 && v36 != v30 && v90 < 0 )
          {
            v79 = a6;
            v86 = v15;
            v47 = sub_2EC06C0(a1[1], v30, byte_3F871B3, 0, v31, a6);
            v48 = *(unsigned __int8 **)(a3 + 80);
            v49 = v86;
            v50 = v47;
            v51 = a1[2];
            v52 = v79;
            v92 = v48;
            v53 = *(_QWORD *)(v51 + 8) - 800LL;
            if ( v48 )
            {
              v75 = v79;
              v80 = *(_QWORD *)(v51 + 8) - 800LL;
              sub_B96E90((__int64)&v92, (__int64)v48, 1);
              v53 = v80;
              v49 = v86;
              v93[0] = (__int64)v92;
              v52 = v75;
              if ( v92 )
              {
                v81 = v86;
                v87 = v53;
                sub_B976B0((__int64)&v92, v92, (__int64)v93);
                v52 = v75;
                v92 = 0;
                v49 = v81;
                v53 = v87;
              }
            }
            else
            {
              v93[0] = 0;
            }
            v54 = (__int64 *)a1[6];
            v55 = a1[5];
            v82 = v52;
            v88 = v49;
            v93[1] = 0;
            v93[2] = 0;
            v56 = sub_2F26260(v55, v54, v93, v53, v50);
            v94.m128i_i64[0] = 0;
            v95 = 0;
            v94.m128i_i32[2] = v90;
            v96 = 0;
            v97 = 0;
            sub_2E8EAD0(v57, (__int64)v56, &v94);
            v15 = v88;
            a6 = v82;
            if ( v93[0] )
            {
              sub_B91220((__int64)v93, v93[0]);
              a6 = v82;
              v15 = v88;
            }
            if ( v92 )
            {
              v89 = a6;
              v91 = v15;
              sub_B91220((__int64)&v92, (__int64)v92);
              a6 = v89;
              v15 = v91;
            }
            v90 = v50;
          }
          if ( !a6 )
            goto LABEL_32;
          goto LABEL_30;
        }
        if ( v29 )
        {
          v34 = a1[3];
          v33 = 0;
          v35 = *(__int64 (**)())(*(_QWORD *)v34 + 176LL);
          if ( v35 != sub_2FF51F0 )
          {
            v74 = a6;
            v76 = a1[4];
            v64 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v35)(v34, v30, 0);
            v26 = v78;
            v31 = v76;
            v15 = 0;
            a6 = v74;
            v33 = v64;
            v32 = (unsigned __int16)v78;
          }
          goto LABEL_21;
        }
      }
      else
      {
        if ( !(_WORD)v26 )
          goto LABEL_32;
        v31 = a1[4];
        v32 = (unsigned __int16)v26;
        if ( !*(_QWORD *)(v31 + 8LL * (unsigned __int16)v26 + 112) )
          goto LABEL_32;
        v30 = 0;
        v33 = 1;
        v85 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v31 + 552LL);
        if ( (*(_BYTE *)(a3 + 32) & 4) != 0 )
          goto LABEL_21;
      }
      v33 = 0;
      goto LABEL_21;
    }
    if ( v14 == 10 )
    {
      v94.m128i_i64[0] = 12;
      v16 = *(_QWORD *)(a3 + 96);
      v17 = a2[1];
      goto LABEL_6;
    }
    if ( (unsigned int)(v14 - 37) <= 1 || (unsigned int)(v14 - 13) <= 1 )
    {
      v39 = *(_QWORD *)(a3 + 96);
      v40 = *(_DWORD *)(a3 + 112);
      v94.m128i_i8[0] = 10;
      v41 = *(_QWORD *)(a3 + 104);
      v42 = a2[1];
      v95 = 0;
      v96 = v39;
LABEL_36:
      LODWORD(v97) = HIDWORD(v41);
      v94.m128i_i32[2] = v41;
LABEL_37:
      v43 = *a2;
      v94.m128i_i32[0] = ((v40 & 0xFFF) << 8) | v94.m128i_i32[0] & 0xFFF000FF;
      return sub_2E8EAD0(v42, v43, &v94);
    }
    if ( v14 == 6 )
    {
      v44 = a2[1];
      v45 = *a2;
      v94.m128i_i8[0] = 4;
      v46 = *(_QWORD *)(a3 + 96);
      v95 = 0;
      v94.m128i_i32[0] &= 0xFFF000FF;
      v96 = v46;
      return sub_2E8EAD0(v44, v45, &v94);
    }
    if ( v14 != 39 && v14 != 15 )
    {
      switch ( v14 )
      {
        case 40:
        case 16:
          v65 = *(_DWORD *)(a3 + 96);
          v40 = *(_DWORD *)(a3 + 100);
          v94.m128i_i8[0] = 8;
          v95 = 0;
          v42 = a2[1];
          LODWORD(v96) = v65;
          goto LABEL_37;
        case 17:
        case 41:
          v66 = *(unsigned __int8 *)(a3 + 108);
          v67 = *(_QWORD *)(a3 + 96);
          v68 = *(_QWORD *)(*a1 + 56);
          v69 = *(_DWORD *)(a3 + 104) & 0x7FFFFFFF;
          if ( *(int *)(a3 + 104) < 0 )
            v40 = sub_2E7FCE0(v68, v67, v66);
          else
            v40 = sub_2E7C7D0(v68, v67, v66);
          v70 = *(_DWORD *)(a3 + 112);
          LODWORD(v96) = v40;
          v94.m128i_i8[0] = 6;
          v42 = a2[1];
          v95 = 0;
          LOWORD(v40) = v70;
          v94.m128i_i32[2] = v69;
          LODWORD(v97) = 0;
          goto LABEL_37;
        case 18:
        case 42:
          v71 = *(_QWORD *)(a3 + 96);
          v94.m128i_i8[0] = 9;
          v40 = *(_DWORD *)(a3 + 104);
          v42 = a2[1];
          v94.m128i_i32[2] = 0;
          v95 = 0;
          v96 = v71;
          LODWORD(v97) = 0;
          goto LABEL_37;
        case 44:
          v58 = a2[1];
          v59 = *a2;
          v94.m128i_i8[0] = 15;
          v60 = *(_QWORD *)(a3 + 96);
          v95 = 0;
          v94.m128i_i32[0] &= 0xFFF000FF;
          v96 = v60;
          v94.m128i_i32[2] = 0;
          LODWORD(v97) = 0;
          return sub_2E8EAD0(v58, v59, &v94);
        case 43:
        case 19:
          v73 = *(_QWORD *)(a3 + 96);
          v40 = *(_DWORD *)(a3 + 112);
          v94.m128i_i8[0] = 11;
          v41 = *(_QWORD *)(a3 + 104);
          v42 = a2[1];
          v95 = 0;
          v96 = v73;
          goto LABEL_36;
        case 45:
          v72 = *(_DWORD *)(a3 + 100);
          v40 = *(_DWORD *)(a3 + 96);
          v94.m128i_i8[0] = 7;
          v95 = 0;
          v41 = *(_QWORD *)(a3 + 104);
          LODWORD(v96) = v72;
          v42 = a2[1];
          goto LABEL_36;
      }
      return sub_3752220(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
    }
    v61 = *(_DWORD *)(a3 + 96);
    v62 = a2[1];
    v94.m128i_i64[0] = 5;
    v63 = *a2;
    v95 = 0;
    LODWORD(v96) = v61;
    return sub_2E8EAD0(v62, v63, &v94);
  }
}
