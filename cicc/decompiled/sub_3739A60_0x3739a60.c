// Function: sub_3739A60
// Address: 0x3739a60
//
void __fastcall sub_3739A60(__int64 *a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5)
{
  __int64 *v6; // rbx
  unsigned __int64 v7; // r13
  __int64 v8; // rdi
  int v9; // eax
  const char *v10; // rdx
  int v11; // ecx
  char v12; // al
  __m128i *v13; // r14
  __int64 v14; // r12
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rsi
  unsigned __int64 v18; // rdi
  unsigned __int64 *v19; // rdi
  __m128i *v20; // rsi
  unsigned __int64 v21; // rdi
  _BOOL8 v22; // rcx
  __int64 v23; // rax
  __int64 v24; // rcx
  __int16 v25; // r8
  __int64 (*v26)(); // rdx
  __int64 v27; // rsi
  __int64 v28; // rdi
  __int64 (*v29)(); // rax
  __int64 v30; // rdi
  int v31; // eax
  unsigned int v32; // eax
  __int64 v33; // rax
  _BOOL8 v34; // rcx
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // r12
  unsigned __int8 v38; // al
  __int64 v39; // rbx
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // rdx
  __int64 v43; // r8
  unsigned __int8 v44; // al
  __int64 v45; // rdx
  __int64 v46; // rdi
  __int64 v47; // rdx
  unsigned __int8 v48; // al
  __int64 v49; // rax
  const void *v50; // r12
  size_t v51; // rdx
  size_t v52; // r14
  __int64 v53; // rdx
  __int64 v54; // rdi
  const void *v55; // rax
  __int64 v56; // rdx
  __int64 v57; // r12
  unsigned __int8 v58; // al
  __int64 v59; // rbx
  __int64 v60; // rcx
  __int64 v61; // rdx
  __int64 v62; // r8
  __int64 v63; // rax
  unsigned __int8 v64; // al
  __int64 v65; // rdx
  _BYTE *v66; // rdx
  __int64 v67; // rax
  size_t v68; // rdx
  size_t v69; // rcx
  __int64 v70; // r8
  __int64 v71; // r12
  __int64 v72; // rax
  _BOOL4 v73; // [rsp+Ch] [rbp-74h]
  __int16 v74; // [rsp+Ch] [rbp-74h]
  unsigned __int32 v77; // [rsp+20h] [rbp-60h]
  char v78; // [rsp+25h] [rbp-5Bh]
  char v79; // [rsp+26h] [rbp-5Ah]
  bool v80; // [rsp+27h] [rbp-59h]
  __int64 v81; // [rsp+28h] [rbp-58h]
  __int64 v82; // [rsp+28h] [rbp-58h]
  __int64 v83; // [rsp+28h] [rbp-58h]
  __int64 v84; // [rsp+28h] [rbp-58h]
  __int64 v85; // [rsp+28h] [rbp-58h]
  __int64 *v86; // [rsp+30h] [rbp-50h]
  __int8 v87; // [rsp+30h] [rbp-50h]
  unsigned __int64 **v88; // [rsp+38h] [rbp-48h]
  __m128i v89[4]; // [rsp+40h] [rbp-40h] BYREF

  v86 = &a4[2 * a5];
  if ( a4 == v86 )
  {
    if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1[23] + 200) + 544LL) - 42) <= 1 )
    {
      v63 = a1[26];
      if ( *(_DWORD *)(v63 + 6224) == 1 )
      {
        v79 = 0;
        v7 = 0;
        v70 = 5;
        v88 = 0;
LABEL_118:
        v89[0].m128i_i32[0] = 65547;
        sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 51, 65547, v70);
        goto LABEL_56;
      }
      if ( !*(_BYTE *)(v63 + 3686) )
        return;
    }
    else if ( !*(_BYTE *)(a1[26] + 3686) )
    {
      return;
    }
    v79 = 0;
    v7 = 0;
LABEL_110:
    v64 = *(_BYTE *)(a3 - 16);
    if ( (v64 & 2) != 0 )
      v65 = *(_QWORD *)(a3 - 32);
    else
      v65 = a3 - 16 - 8LL * ((v64 >> 2) & 0xF);
    v66 = *(_BYTE **)(v65 + 40);
    if ( v66 )
    {
      v67 = sub_B91420((__int64)v66);
      v69 = v68;
      v66 = (_BYTE *)v67;
    }
    else
    {
      v69 = 0;
    }
    sub_324B070(a1, a2, v66, v69);
    goto LABEL_59;
  }
  v88 = 0;
  v6 = a4;
  v80 = a5 == 1;
  v7 = 0;
  v79 = 0;
  v78 = 0;
  v77 = 0;
  while ( 1 )
  {
    while ( 1 )
    {
      v13 = (__m128i *)v6[1];
      v14 = *v6;
      if ( v13 )
      {
        if ( v80 )
          break;
      }
      if ( !v14 )
      {
        if ( !v13 )
          goto LABEL_23;
        goto LABEL_28;
      }
LABEL_3:
      if ( (*(_BYTE *)(v14 + 33) & 3) == 1 )
        goto LABEL_23;
      if ( (*(_BYTE *)(v14 + 33) & 0x1C) == 0 || *(_BYTE *)(sub_31DA6B0(a1[23]) + 938) )
      {
        if ( v88 )
          goto LABEL_6;
LABEL_30:
        v15 = sub_A777F0(0x10u, a1 + 11);
        v88 = (unsigned __int64 **)v15;
        if ( v15 )
        {
          *(_QWORD *)v15 = 0;
          *(_DWORD *)(v15 + 8) = 0;
        }
        v82 = a1[23];
        v16 = sub_22077B0(0x90u);
        if ( v16 )
        {
          v17 = v82;
          v83 = v16;
          sub_3247620(v16, v17, (__int64)a1, (__int64)v88);
          v16 = v83;
        }
        if ( v7 )
        {
          v18 = *(_QWORD *)(v7 + 24);
          if ( v18 != v7 + 40 )
          {
            v84 = v16;
            _libc_free(v18);
            v16 = v84;
          }
          v85 = v16;
          j_j___libc_free_0(v7);
          v16 = v85;
        }
        v79 = 1;
        v7 = v16;
LABEL_6:
        if ( v13 )
        {
          if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1[23] + 200) + 544LL) - 42) <= 1
            && *(_DWORD *)(a1[26] + 6224) == 1 )
          {
            v33 = sub_B0D5A0((__int64)v13, v89);
            if ( v13 != (__m128i *)v33 )
            {
              v78 = 1;
              v77 = v89[0].m128i_i32[0];
            }
            v13 = (__m128i *)v33;
          }
          sub_3243D60((_QWORD *)v7, (__int64)v13);
        }
        if ( !v14 )
          goto LABEL_18;
        v81 = sub_31DB510(a1[23], v14);
        v8 = *(_QWORD *)(a1[23] + 200);
        v9 = *(_DWORD *)(v8 + 544);
        if ( (*(_BYTE *)(v14 + 33) & 0x1C) != 0 )
        {
          if ( (unsigned int)(v9 - 56) <= 1 )
          {
            v10 = "__tls_base";
            v11 = 10;
            goto LABEL_14;
          }
          if ( !sub_23CF310(v8) )
          {
            if ( *(_BYTE *)(a1[26] + 3769) )
            {
              sub_3249B00(a1, v88, 11, 252);
              v32 = sub_37291A0(a1[26] + 4840, v81, 1);
              sub_3249B00(a1, v88, 15, v32);
            }
            else
            {
              v34 = *(_DWORD *)(*(_QWORD *)(a1[23] + 208) + 8LL) != 4;
              v74 = v34 + 6;
              sub_3249B00(a1, v88, 11, 2 * v34 + 12);
              v35 = sub_31DA6B0(a1[23]);
              v36 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v35 + 168LL))(v35, v81);
              sub_37378A0((__int64)a1, v88, v74, v36);
            }
            sub_3249B00(a1, v88, 11, (-(__int64)(*(_BYTE *)(a1[26] + 3684) == 0) & 0xFFFFFFFFFFFFFFBBLL) + 224);
          }
LABEL_16:
          if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1[23] + 200) + 544LL) - 42) <= 1
            && *(_DWORD *)(a1[26] + 6224) == 1
            && !v78 )
          {
            if ( *(_DWORD *)(*(_QWORD *)(v14 + 8) + 8LL) <= 0x5FFu )
            {
              switch ( *(_DWORD *)(*(_QWORD *)(v14 + 8) + 8LL) >> 8 )
              {
                case 0:
                  v78 = 1;
                  v77 = 12;
                  goto LABEL_18;
                case 1:
                  v78 = 1;
                  v77 = 5;
                  goto LABEL_18;
                case 2:
                  break;
                case 3:
                  v78 = 1;
                  v77 = 8;
                  goto LABEL_18;
                case 4:
                  v78 = 1;
                  v77 = 4;
                  goto LABEL_18;
                case 5:
                  v78 = 1;
                  v77 = 6;
                  goto LABEL_18;
              }
            }
            BUG();
          }
LABEL_18:
          v12 = *(_BYTE *)(v7 + 100);
          if ( (v12 & 7) == 0 )
            *(_BYTE *)(v7 + 100) = v12 & 0xF8 | 2;
          v89[0] = 0u;
          if ( v13 )
            v89[0] = v13[1];
          sub_3244870((_BYTE *)v7, (unsigned __int64 **)v89);
          goto LABEL_23;
        }
        if ( (unsigned int)(v9 - 56) > 1 )
        {
LABEL_45:
          if ( (unsigned int)sub_23CF1A0(v8) != 4 && (unsigned int)sub_23CF1A0(*(_QWORD *)(a1[23] + 200)) != 5
            || (sub_31DA6B0(a1[23]), (unsigned __int8)(sub_31578C0(v14, *(_QWORD *)(a1[23] + 200)) - 4) <= 7u) )
          {
            v19 = (unsigned __int64 *)a1[26];
            v89[0].m128i_i64[1] = (__int64)a1;
            v89[0].m128i_i64[0] = v81;
            v20 = (__m128i *)v19[89];
            if ( v20 == (__m128i *)v19[90] )
            {
              sub_3223B70(v19 + 88, v20, v89);
            }
            else
            {
              if ( v20 )
              {
                *v20 = _mm_loadu_si128(v89);
                v20 = (__m128i *)v19[89];
              }
              v19[89] = (unsigned __int64)&v20[1];
            }
            sub_324BB60(a1, v88, v81);
            goto LABEL_16;
          }
          v22 = *(_DWORD *)(*(_QWORD *)(a1[23] + 208) + 8LL) != 4;
          v73 = v22 + 6;
          sub_3249B00(a1, v88, 11, 2 * v22 + 12);
          v23 = sub_31DA6B0(a1[23]);
          v24 = 0;
          v25 = v73;
          v26 = *(__int64 (**)())(*(_QWORD *)v23 + 200LL);
          if ( v26 != sub_302E470 )
          {
            v72 = ((__int64 (__fastcall *)(__int64, __int64, __int64 (*)(), _QWORD, _BOOL4))v26)(v23, v81, v26, 0, v73);
            v25 = v73;
            v24 = v72;
          }
          sub_37378A0((__int64)a1, v88, v25, v24);
          v27 = 0;
          v28 = sub_31DA6B0(a1[23]);
          v29 = *(__int64 (**)())(*(_QWORD *)v28 + 192LL);
          if ( v29 != sub_302E460 )
            v27 = ((unsigned int (__fastcall *)(__int64, _QWORD))v29)(v28, 0);
          v30 = *(_QWORD *)(*(_QWORD *)(a1[23] + 200) + 664LL);
          v31 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v30 + 16LL))(v30, v27, 0);
          sub_3249B00(a1, v88, 11, (unsigned int)(v31 + 112));
          v89[0].m128i_i32[0] = 65549;
          sub_32499D0(a1, v88, 65549, 0);
        }
        else
        {
          if ( (unsigned int)sub_23CF1A0(v8) != 1 )
          {
            v8 = *(_QWORD *)(a1[23] + 200);
            goto LABEL_45;
          }
          v10 = "__memory_base";
          v11 = 13;
LABEL_14:
          sub_3735D90(a1, v88, (char)v10, v11, 1);
          sub_324BB60(a1, v88, v81);
        }
        sub_3249B00(a1, v88, 11, 34);
        goto LABEL_16;
      }
      v6 += 2;
      if ( v86 == v6 )
        goto LABEL_54;
    }
    v89[0].m128i_i64[0] = sub_AF4F20(v6[1]);
    if ( v89[0].m128i_i8[4] )
      break;
    if ( v14 )
      goto LABEL_3;
LABEL_28:
    v89[0].m128i_i64[0] = sub_AF4F20((__int64)v13);
    if ( v89[0].m128i_i8[4] )
    {
      if ( !v88 )
        goto LABEL_30;
      goto LABEL_6;
    }
LABEL_23:
    v6 += 2;
    if ( v86 == v6 )
      goto LABEL_54;
  }
  v87 = v89[0].m128i_i8[4];
  v71 = *(_QWORD *)(v13[1].m128i_i64[0] + 8);
  v89[0].m128i_i64[0] = sub_AF4F20((__int64)v13);
  sub_3249E90(a1, a2, v89[0].m128i_i32[0] == 1, v71);
  v79 = v87;
LABEL_54:
  if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1[23] + 200) + 544LL) - 42) <= 1 && *(_DWORD *)(a1[26] + 6224) == 1 )
  {
    v70 = v77;
    if ( !v78 )
      v70 = 5;
    goto LABEL_118;
  }
LABEL_56:
  if ( v88 )
  {
    sub_3243D40(v7);
    sub_3249620(a1, a2, 2, *(__int64 ***)(v7 + 112));
  }
  if ( *(_BYTE *)(a1[26] + 3686) )
    goto LABEL_110;
LABEL_59:
  if ( !v79 )
    goto LABEL_60;
  v37 = a1[26];
  v38 = *(_BYTE *)(a3 - 16);
  v39 = a3 - 16;
  v40 = (v38 & 2) != 0 ? *(_QWORD *)(a3 - 32) : v39 - 8LL * ((v38 >> 2) & 0xF);
  v41 = *(_QWORD *)(v40 + 8);
  if ( v41 )
  {
    v41 = sub_B91420(*(_QWORD *)(v40 + 8));
    v43 = v42;
  }
  else
  {
    v43 = 0;
  }
  sub_3237930(v37, (__int64)a1, *(_DWORD *)(a1[10] + 36), v41, v43, a2);
  v44 = *(_BYTE *)(a3 - 16);
  v45 = (v44 & 2) != 0 ? *(_QWORD *)(a3 - 32) : v39 - 8LL * ((v44 >> 2) & 0xF);
  v46 = *(_QWORD *)(v45 + 40);
  if ( !v46 )
    goto LABEL_60;
  sub_B91420(v46);
  if ( !v47 )
    goto LABEL_60;
  v48 = *(_BYTE *)(a3 - 16);
  if ( (v48 & 2) != 0 )
  {
    v49 = *(_QWORD *)(a3 - 32);
    v50 = *(const void **)(v49 + 40);
    if ( v50 )
      goto LABEL_95;
    v54 = *(_QWORD *)(v49 + 8);
    if ( v54 )
    {
      v52 = 0;
      goto LABEL_98;
    }
    goto LABEL_60;
  }
  v50 = *(const void **)(v39 - 8LL * ((v48 >> 2) & 0xF) + 40);
  if ( v50 )
  {
LABEL_95:
    v50 = (const void *)sub_B91420((__int64)v50);
    v52 = v51;
    v48 = *(_BYTE *)(a3 - 16);
    if ( (v48 & 2) == 0 )
      goto LABEL_96;
    v53 = *(_QWORD *)(a3 - 32);
  }
  else
  {
    v52 = 0;
LABEL_96:
    v53 = v39 - 8LL * ((v48 >> 2) & 0xF);
  }
  v54 = *(_QWORD *)(v53 + 8);
  if ( v54 )
  {
LABEL_98:
    v55 = (const void *)sub_B91420(v54);
    if ( v52 == v56 && (!v52 || !memcmp(v55, v50, v52)) )
      goto LABEL_60;
  }
  else if ( !v52 )
  {
    goto LABEL_60;
  }
  v57 = a1[26];
  if ( *(_BYTE *)(v57 + 3686) )
  {
    v58 = *(_BYTE *)(a3 - 16);
    if ( (v58 & 2) != 0 )
      v59 = *(_QWORD *)(a3 - 32);
    else
      v59 = v39 - 8LL * ((v58 >> 2) & 0xF);
    v60 = *(_QWORD *)(v59 + 40);
    if ( v60 )
    {
      v60 = sub_B91420(*(_QWORD *)(v59 + 40));
      v62 = v61;
    }
    else
    {
      v62 = 0;
    }
    sub_3237930(v57, (__int64)a1, *(_DWORD *)(a1[10] + 36), v60, v62, a2);
  }
LABEL_60:
  if ( v7 )
  {
    v21 = *(_QWORD *)(v7 + 24);
    if ( v21 != v7 + 40 )
      _libc_free(v21);
    j_j___libc_free_0(v7);
  }
}
