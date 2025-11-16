// Function: sub_C5DD90
// Address: 0xc5dd90
//
__int64 __fastcall sub_C5DD90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  _QWORD *v5; // rax
  __int64 v6; // rdx
  _QWORD *v7; // r13
  _BYTE *i; // rsi
  _QWORD *v9; // rbx
  __int64 v10; // rsi
  __int64 v11; // r13
  _QWORD *v12; // r14
  __int64 *v13; // rbx
  __int64 *v14; // r12
  _QWORD *v15; // rdi
  int v16; // r11d
  unsigned int v17; // edx
  _QWORD *v18; // rax
  __int64 v19; // r8
  _QWORD *v20; // rdx
  _BYTE *v21; // rsi
  _QWORD *v22; // rdi
  unsigned int v23; // ecx
  int v24; // eax
  __int64 v25; // r8
  __int64 v26; // rax
  __int64 *v27; // r14
  __int64 *v28; // r13
  __int64 v29; // rsi
  __int64 v30; // r15
  int v31; // r10d
  _QWORD *v32; // rax
  __int64 v33; // rdi
  _QWORD *v34; // rbx
  __int64 v35; // rcx
  __int64 v36; // rdi
  _BYTE *v37; // rax
  __int64 v38; // rax
  size_t v39; // rdx
  const void *v40; // rsi
  _WORD *v41; // rdi
  __int64 v42; // r8
  unsigned __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 *v46; // r15
  __int64 *v47; // rbx
  __int64 v48; // rdi
  unsigned int v49; // eax
  _QWORD *v50; // r12
  _QWORD *v51; // rbx
  __int64 v52; // rdi
  __int64 result; // rax
  int v54; // r11d
  _QWORD *v55; // r10
  unsigned int v56; // edx
  __int64 v57; // r8
  int v58; // edi
  __int64 v59; // rax
  __int64 v60; // rax
  _QWORD *v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rax
  unsigned int v64; // edx
  __int64 v65; // rcx
  int v66; // r10d
  _QWORD *v67; // r9
  int v68; // r10d
  unsigned int v69; // edx
  __int64 v70; // rsi
  int v71; // r11d
  __int64 v72; // [rsp+8h] [rbp-98h]
  __int64 v74; // [rsp+10h] [rbp-90h]
  __int64 v75; // [rsp+10h] [rbp-90h]
  __int64 v76; // [rsp+18h] [rbp-88h]
  __int64 v77; // [rsp+28h] [rbp-78h] BYREF
  void *base; // [rsp+30h] [rbp-70h] BYREF
  _BYTE *v79; // [rsp+38h] [rbp-68h]
  _BYTE *v80; // [rsp+40h] [rbp-60h]
  __int64 v81; // [rsp+50h] [rbp-50h] BYREF
  _QWORD *v82; // [rsp+58h] [rbp-48h]
  __int64 v83; // [rsp+60h] [rbp-40h]
  unsigned int v84; // [rsp+68h] [rbp-38h]

  base = 0;
  v79 = 0;
  v80 = 0;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  v4 = sub_C4F9D0(a1, a2);
  v5 = *(_QWORD **)(v4 + 128);
  if ( *(_BYTE *)(v4 + 148) )
    v6 = *(unsigned int *)(v4 + 140);
  else
    v6 = *(unsigned int *)(v4 + 136);
  v7 = &v5[v6];
  i = v79;
  if ( v5 != v7 )
  {
    while ( 1 )
    {
      v9 = v5;
      if ( *v5 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v7 == ++v5 )
        goto LABEL_6;
    }
    if ( v7 != v5 )
    {
      v77 = *v5;
LABEL_99:
      sub_C5D3E0((__int64)&base, i, &v77);
      for ( i = v79; ; v79 = i )
      {
        v61 = v9 + 1;
        if ( v9 + 1 == v7 )
          break;
        v62 = *v61;
        for ( ++v9; *v61 >= 0xFFFFFFFFFFFFFFFELL; v9 = v61 )
        {
          if ( v7 == ++v61 )
            goto LABEL_6;
          v62 = *v61;
        }
        if ( v7 == v9 )
          break;
        v77 = v62;
        if ( v80 == i )
          goto LABEL_99;
        if ( i )
        {
          *(_QWORD *)i = v62;
          i = v79;
        }
        i += 8;
      }
    }
  }
LABEL_6:
  v10 = i - (_BYTE *)base;
  if ( v10 > 8 )
    qsort(base, v10 >> 3, 8u, (__compar_fn_t)sub_C4FAE0);
  v11 = 0;
  v76 = *(unsigned int *)(a2 + 8);
  if ( *(_DWORD *)(a2 + 8) )
  {
    v12 = (_QWORD *)a2;
    v74 = a3;
    while ( 1 )
    {
      v77 = *(_QWORD *)(*v12 + 16 * v11 + 8);
      v13 = *(__int64 **)(v77 + 72);
      v14 = &v13[*(unsigned int *)(v77 + 80)];
      if ( v14 != v13 )
        break;
LABEL_25:
      if ( v76 == ++v11 )
      {
        a3 = v74;
        goto LABEL_27;
      }
    }
    while ( v84 )
    {
      v15 = 0;
      v16 = 1;
      v17 = (v84 - 1) & (((unsigned int)*v13 >> 9) ^ ((unsigned int)*v13 >> 4));
      v18 = &v82[4 * v17];
      v19 = *v18;
      if ( *v13 != *v18 )
      {
        while ( v19 != -4096 )
        {
          if ( v19 == -8192 && !v15 )
            v15 = v18;
          v17 = (v84 - 1) & (v16 + v17);
          v18 = &v82[4 * v17];
          v19 = *v18;
          if ( *v13 == *v18 )
            goto LABEL_13;
          ++v16;
        }
        if ( !v15 )
          v15 = v18;
        ++v81;
        v24 = v83 + 1;
        if ( 4 * ((int)v83 + 1) < 3 * v84 )
        {
          if ( v84 - HIDWORD(v83) - v24 <= v84 >> 3 )
          {
            sub_C5DB60((__int64)&v81, v84);
            if ( !v84 )
            {
LABEL_136:
              LODWORD(v83) = v83 + 1;
              BUG();
            }
            v54 = 1;
            v55 = 0;
            v56 = (v84 - 1) & (((unsigned int)*v13 >> 9) ^ ((unsigned int)*v13 >> 4));
            v24 = v83 + 1;
            v15 = &v82[4 * v56];
            v57 = *v15;
            if ( *v13 != *v15 )
            {
              while ( v57 != -4096 )
              {
                if ( !v55 && v57 == -8192 )
                  v55 = v15;
                v56 = (v84 - 1) & (v54 + v56);
                v15 = &v82[4 * v56];
                v57 = *v15;
                if ( *v13 == *v15 )
                  goto LABEL_21;
                ++v54;
              }
LABEL_68:
              if ( v55 )
                v15 = v55;
            }
          }
LABEL_21:
          LODWORD(v83) = v24;
          if ( *v15 != -4096 )
            --HIDWORD(v83);
          v26 = *v13;
          v22 = v15 + 1;
          v21 = 0;
          *v22 = 0;
          v22[1] = 0;
          *(v22 - 1) = v26;
          v22[2] = 0;
          goto LABEL_24;
        }
LABEL_19:
        sub_C5DB60((__int64)&v81, 2 * v84);
        if ( !v84 )
          goto LABEL_136;
        v23 = (v84 - 1) & (((unsigned int)*v13 >> 9) ^ ((unsigned int)*v13 >> 4));
        v24 = v83 + 1;
        v15 = &v82[4 * v23];
        v25 = *v15;
        if ( *v15 != *v13 )
        {
          v71 = 1;
          v55 = 0;
          while ( v25 != -4096 )
          {
            if ( v55 || v25 != -8192 )
              v15 = v55;
            v23 = (v84 - 1) & (v71 + v23);
            v25 = v82[4 * v23];
            if ( *v13 == v25 )
            {
              v15 = &v82[4 * v23];
              goto LABEL_21;
            }
            ++v71;
            v55 = v15;
            v15 = &v82[4 * v23];
          }
          goto LABEL_68;
        }
        goto LABEL_21;
      }
LABEL_13:
      v20 = (_QWORD *)v18[2];
      v21 = (_BYTE *)v18[3];
      v22 = v18 + 1;
      if ( v20 == (_QWORD *)v21 )
      {
LABEL_24:
        ++v13;
        sub_C5D570((__int64)v22, v21, &v77);
        if ( v14 == v13 )
          goto LABEL_25;
      }
      else
      {
        if ( v20 )
        {
          *v20 = v77;
          v20 = (_QWORD *)v18[2];
        }
        ++v13;
        v18[2] = v20 + 1;
        if ( v14 == v13 )
          goto LABEL_25;
      }
    }
    ++v81;
    goto LABEL_19;
  }
LABEL_27:
  v27 = (__int64 *)v79;
  v28 = (__int64 *)base;
  v29 = v84;
  if ( v79 != base )
  {
    while ( 1 )
    {
      v30 = *v28;
      if ( !(_DWORD)v29 )
        break;
      v31 = 1;
      v32 = 0;
      v33 = ((_DWORD)v29 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
      v34 = &v82[4 * v33];
      v35 = *v34;
      if ( v30 != *v34 )
      {
        while ( v35 != -4096 )
        {
          if ( v35 == -8192 && !v32 )
            v32 = v34;
          v33 = ((_DWORD)v29 - 1) & (unsigned int)(v31 + v33);
          v34 = &v82[4 * (unsigned int)v33];
          v35 = *v34;
          if ( v30 == *v34 )
            goto LABEL_30;
          ++v31;
        }
        if ( !v32 )
          v32 = v34;
        ++v81;
        v58 = v83 + 1;
        if ( 4 * ((int)v83 + 1) < (unsigned int)(3 * v29) )
        {
          if ( (int)v29 - HIDWORD(v83) - v58 <= (unsigned int)v29 >> 3 )
          {
            sub_C5DB60((__int64)&v81, v29);
            if ( !v84 )
            {
LABEL_135:
              LODWORD(v83) = v83 + 1;
              BUG();
            }
            v67 = 0;
            v68 = 1;
            v69 = (v84 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
            v58 = v83 + 1;
            v32 = &v82[4 * v69];
            v70 = *v32;
            if ( v30 != *v32 )
            {
              while ( v70 != -4096 )
              {
                if ( v70 == -8192 && !v67 )
                  v67 = v32;
                v69 = (v84 - 1) & (v68 + v69);
                v32 = &v82[4 * v69];
                v70 = *v32;
                if ( v30 == *v32 )
                  goto LABEL_81;
                ++v68;
              }
              goto LABEL_114;
            }
          }
          goto LABEL_81;
        }
LABEL_102:
        sub_C5DB60((__int64)&v81, 2 * v29);
        if ( !v84 )
          goto LABEL_135;
        v64 = (v84 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
        v58 = v83 + 1;
        v32 = &v82[4 * v64];
        v65 = *v32;
        if ( v30 != *v32 )
        {
          v66 = 1;
          v67 = 0;
          while ( v65 != -4096 )
          {
            if ( !v67 && v65 == -8192 )
              v67 = v32;
            v64 = (v84 - 1) & (v66 + v64);
            v32 = &v82[4 * v64];
            v65 = *v32;
            if ( v30 == *v32 )
              goto LABEL_81;
            ++v66;
          }
LABEL_114:
          if ( v67 )
            v32 = v67;
        }
LABEL_81:
        LODWORD(v83) = v58;
        if ( *v32 != -4096 )
          --HIDWORD(v83);
        *v32 = v30;
        v32[1] = 0;
        v32[2] = 0;
        v32[3] = 0;
        goto LABEL_42;
      }
LABEL_30:
      if ( v34[1] != v34[2] )
      {
        v36 = sub_CB7210(v33);
        v37 = *(_BYTE **)(v36 + 32);
        if ( *(_BYTE **)(v36 + 24) == v37 )
        {
          sub_CB6200(v36, "\n", 1);
        }
        else
        {
          *v37 = 10;
          ++*(_QWORD *)(v36 + 32);
        }
        v38 = sub_CB7210(v36);
        v39 = *(_QWORD *)(v30 + 8);
        v40 = *(const void **)v30;
        v41 = *(_WORD **)(v38 + 32);
        v42 = v38;
        v43 = *(_QWORD *)(v38 + 24) - (_QWORD)v41;
        if ( v39 > v43 )
        {
          v60 = sub_CB6200(v42, v40, v39);
          v41 = *(_WORD **)(v60 + 32);
          v42 = v60;
          v43 = *(_QWORD *)(v60 + 24) - (_QWORD)v41;
        }
        else if ( v39 )
        {
          v72 = v42;
          v75 = *(_QWORD *)(v30 + 8);
          memcpy(v41, v40, v39);
          v42 = v72;
          v63 = *(_QWORD *)(v72 + 24);
          v41 = (_WORD *)(*(_QWORD *)(v72 + 32) + v75);
          *(_QWORD *)(v72 + 32) = v41;
          v43 = v63 - (_QWORD)v41;
        }
        if ( v43 <= 1 )
        {
          v41 = (_WORD *)v42;
          sub_CB6200(v42, ":\n", 2);
        }
        else
        {
          *v41 = 2618;
          *(_QWORD *)(v42 + 32) += 2LL;
        }
        if ( *(_QWORD *)(v30 + 24) )
        {
          v44 = sub_CB7210(v41);
          v45 = sub_A51340(v44, *(const void **)(v30 + 16), *(_QWORD *)(v30 + 24));
          sub_904010(v45, "\n\n");
        }
        else
        {
          v59 = sub_CB7210(v41);
          sub_904010(v59, "\n");
        }
        v46 = (__int64 *)v34[1];
        v47 = (__int64 *)v34[2];
        while ( v47 != v46 )
        {
          v48 = *v46++;
          (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v48 + 48LL))(v48, a3);
        }
LABEL_42:
        v29 = v84;
      }
      if ( v27 == ++v28 )
        goto LABEL_44;
    }
    ++v81;
    goto LABEL_102;
  }
LABEL_44:
  v49 = v29;
  if ( (_DWORD)v29 )
  {
    v50 = v82;
    v51 = &v82[4 * v29];
    do
    {
      if ( *v50 != -8192 && *v50 != -4096 )
      {
        v52 = v50[1];
        if ( v52 )
          j_j___libc_free_0(v52, v50[3] - v52);
      }
      v50 += 4;
    }
    while ( v51 != v50 );
    v49 = v84;
  }
  result = sub_C7D6A0(v82, 32LL * v49, 8);
  if ( base )
    return j_j___libc_free_0(base, v80 - (_BYTE *)base);
  return result;
}
