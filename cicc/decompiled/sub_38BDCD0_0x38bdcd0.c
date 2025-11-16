// Function: sub_38BDCD0
// Address: 0x38bdcd0
//
__int64 __fastcall sub_38BDCD0(__int64 a1, __int64 a2)
{
  char *v4; // rbx
  __int64 v5; // r14
  __int64 result; // rax
  char *v7; // r14
  __int64 (*v8)(); // rax
  __int64 (*v9)(); // rax
  __int64 (*v10)(); // rax
  __int64 (*v11)(); // rax
  int v12; // eax
  __int64 v13; // rsi
  int v14; // ecx
  __int64 v15; // r9
  unsigned int v16; // eax
  __int64 *v17; // rdi
  __int64 v18; // r8
  char *v19; // r14
  __int64 v20; // rsi
  int v21; // eax
  int v22; // ecx
  __int64 v23; // r9
  unsigned int v24; // eax
  __int64 *v25; // rdi
  __int64 v26; // r8
  int v27; // eax
  __int64 v28; // rsi
  int v29; // ecx
  __int64 v30; // r9
  unsigned int v31; // eax
  __int64 *v32; // rdi
  __int64 v33; // r8
  int v34; // eax
  __int64 v35; // rsi
  int v36; // ecx
  __int64 v37; // r9
  unsigned int v38; // eax
  __int64 *v39; // rdi
  __int64 v40; // r8
  int v41; // ecx
  __int64 v42; // r8
  __int64 *v43; // rdi
  __int64 v44; // r9
  int v45; // edi
  int v46; // edi
  __int64 v47; // r9
  unsigned int v48; // eax
  __int64 *v49; // rcx
  __int64 v50; // rsi
  int v51; // edi
  int v52; // r10d
  int v53; // edi
  int v54; // r10d
  int v55; // edi
  int v56; // r10d
  int v57; // edi
  int v58; // r10d
  int v59; // edi
  int v60; // r10d
  int v61; // edi
  int v62; // edi
  __int64 v63; // r9
  unsigned int v64; // eax
  __int64 v65; // rsi
  int v66; // ecx
  int v67; // r10d
  int v68; // edi
  int v69; // edi
  __int64 v70; // r9
  unsigned int v71; // eax
  __int64 v72; // rsi
  int v73; // ecx
  int v74; // r10d
  int v75; // ecx
  int v76; // r10d
  char *v77; // [rsp+8h] [rbp-38h]

  v4 = *(char **)(a1 + 1080);
  v77 = *(char **)(a1 + 1088);
  v5 = (v77 - v4) >> 5;
  result = (v77 - v4) >> 3;
  if ( v5 <= 0 )
  {
LABEL_42:
    switch ( result )
    {
      case 2LL:
        result = *(_QWORD *)(*(_QWORD *)a2 + 1056LL);
        break;
      case 3LL:
        result = *(_QWORD *)(*(_QWORD *)a2 + 1056LL);
        if ( (__int64 (*)())result != sub_168DCA0 )
        {
          if ( !((unsigned __int8 (__fastcall *)(__int64, _QWORD))result)(a2, *(_QWORD *)v4) )
          {
            v68 = *(_DWORD *)(a1 + 1072);
            if ( !v68 )
              goto LABEL_16;
            v69 = v68 - 1;
            v70 = *(_QWORD *)(a1 + 1056);
            v71 = v69 & (((unsigned int)*(_QWORD *)v4 >> 9) ^ ((unsigned int)*(_QWORD *)v4 >> 4));
            v49 = (__int64 *)(v70 + 8LL * v71);
            v72 = *v49;
            if ( *v49 != *(_QWORD *)v4 )
            {
              v73 = 1;
              while ( v72 != -8 )
              {
                v74 = v73 + 1;
                v71 = v69 & (v73 + v71);
                v49 = (__int64 *)(v70 + 8LL * v71);
                v72 = *v49;
                if ( *(_QWORD *)v4 == *v49 )
                  goto LABEL_53;
                v73 = v74;
              }
              goto LABEL_16;
            }
            goto LABEL_53;
          }
          result = *(_QWORD *)(*(_QWORD *)a2 + 1056LL);
        }
        v4 += 8;
        break;
      case 1LL:
        result = *(_QWORD *)(*(_QWORD *)a2 + 1056LL);
LABEL_49:
        if ( (__int64 (*)())result == sub_168DCA0 )
          goto LABEL_45;
        result = ((__int64 (__fastcall *)(__int64, _QWORD))result)(a2, *(_QWORD *)v4);
        if ( (_BYTE)result )
          goto LABEL_45;
        v45 = *(_DWORD *)(a1 + 1072);
        if ( !v45 )
          goto LABEL_16;
        v46 = v45 - 1;
        v47 = *(_QWORD *)(a1 + 1056);
        v48 = v46 & (((unsigned int)*(_QWORD *)v4 >> 9) ^ ((unsigned int)*(_QWORD *)v4 >> 4));
        v49 = (__int64 *)(v47 + 8LL * v48);
        v50 = *v49;
        if ( *(_QWORD *)v4 != *v49 )
        {
          v75 = 1;
          while ( v50 != -8 )
          {
            v76 = v75 + 1;
            v48 = v46 & (v75 + v48);
            v49 = (__int64 *)(v47 + 8LL * v48);
            v50 = *v49;
            if ( *(_QWORD *)v4 == *v49 )
              goto LABEL_53;
            v75 = v76;
          }
          goto LABEL_16;
        }
LABEL_53:
        *v49 = -16;
        --*(_DWORD *)(a1 + 1064);
        ++*(_DWORD *)(a1 + 1068);
        goto LABEL_16;
      default:
LABEL_45:
        v4 = v77;
        goto LABEL_27;
    }
    if ( (__int64 (*)())result != sub_168DCA0 )
    {
      if ( !((unsigned __int8 (__fastcall *)(__int64, _QWORD))result)(a2, *(_QWORD *)v4) )
      {
        v61 = *(_DWORD *)(a1 + 1072);
        if ( !v61 )
          goto LABEL_16;
        v62 = v61 - 1;
        v63 = *(_QWORD *)(a1 + 1056);
        v64 = v62 & (((unsigned int)*(_QWORD *)v4 >> 9) ^ ((unsigned int)*(_QWORD *)v4 >> 4));
        v49 = (__int64 *)(v63 + 8LL * v64);
        v65 = *v49;
        if ( *(_QWORD *)v4 != *v49 )
        {
          v66 = 1;
          while ( v65 != -8 )
          {
            v67 = v66 + 1;
            v64 = v62 & (v66 + v64);
            v49 = (__int64 *)(v63 + 8LL * v64);
            v65 = *v49;
            if ( *(_QWORD *)v4 == *v49 )
              goto LABEL_53;
            v66 = v67;
          }
          goto LABEL_16;
        }
        goto LABEL_53;
      }
      result = *(_QWORD *)(*(_QWORD *)a2 + 1056LL);
    }
    v4 += 8;
    goto LABEL_49;
  }
  v7 = &v4[32 * v5];
  while ( 1 )
  {
    v8 = *(__int64 (**)())(*(_QWORD *)a2 + 1056LL);
    if ( v8 != sub_168DCA0 )
      break;
LABEL_3:
    v4 += 32;
    if ( v4 == v7 )
    {
      result = (v77 - v4) >> 3;
      goto LABEL_42;
    }
  }
  if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD))v8)(a2, *(_QWORD *)v4) )
  {
    v9 = *(__int64 (**)())(*(_QWORD *)a2 + 1056LL);
    if ( v9 != sub_168DCA0 )
    {
      if ( !((unsigned __int8 (__fastcall *)(__int64, _QWORD))v9)(a2, *((_QWORD *)v4 + 1)) )
      {
        v27 = *(_DWORD *)(a1 + 1072);
        if ( v27 )
        {
          v28 = *((_QWORD *)v4 + 1);
          v29 = v27 - 1;
          v30 = *(_QWORD *)(a1 + 1056);
          v31 = (v27 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
          v32 = (__int64 *)(v30 + 8LL * v31);
          v33 = *v32;
          if ( v28 == *v32 )
          {
LABEL_32:
            *v32 = -16;
            --*(_DWORD *)(a1 + 1064);
            ++*(_DWORD *)(a1 + 1068);
          }
          else
          {
            v53 = 1;
            while ( v33 != -8 )
            {
              v54 = v53 + 1;
              v31 = v29 & (v53 + v31);
              v32 = (__int64 *)(v30 + 8LL * v31);
              v33 = *v32;
              if ( v28 == *v32 )
                goto LABEL_32;
              v53 = v54;
            }
          }
        }
        v4 += 8;
        goto LABEL_16;
      }
      v10 = *(__int64 (**)())(*(_QWORD *)a2 + 1056LL);
      if ( v10 != sub_168DCA0 )
      {
        if ( !((unsigned __int8 (__fastcall *)(__int64, _QWORD))v10)(a2, *((_QWORD *)v4 + 2)) )
        {
          v34 = *(_DWORD *)(a1 + 1072);
          if ( v34 )
          {
            v35 = *((_QWORD *)v4 + 2);
            v36 = v34 - 1;
            v37 = *(_QWORD *)(a1 + 1056);
            v38 = (v34 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
            v39 = (__int64 *)(v37 + 8LL * v38);
            v40 = *v39;
            if ( *v39 == v35 )
            {
LABEL_36:
              *v39 = -16;
              --*(_DWORD *)(a1 + 1064);
              ++*(_DWORD *)(a1 + 1068);
            }
            else
            {
              v55 = 1;
              while ( v40 != -8 )
              {
                v56 = v55 + 1;
                v38 = v36 & (v55 + v38);
                v39 = (__int64 *)(v37 + 8LL * v38);
                v40 = *v39;
                if ( v35 == *v39 )
                  goto LABEL_36;
                v55 = v56;
              }
            }
          }
          v4 += 16;
          goto LABEL_16;
        }
        v11 = *(__int64 (**)())(*(_QWORD *)a2 + 1056LL);
        if ( v11 != sub_168DCA0 && !((unsigned __int8 (__fastcall *)(__int64, _QWORD))v11)(a2, *((_QWORD *)v4 + 3)) )
        {
          v12 = *(_DWORD *)(a1 + 1072);
          if ( v12 )
          {
            v13 = *((_QWORD *)v4 + 3);
            v14 = v12 - 1;
            v15 = *(_QWORD *)(a1 + 1056);
            v16 = (v12 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
            v17 = (__int64 *)(v15 + 8LL * v16);
            v18 = *v17;
            if ( v13 == *v17 )
            {
LABEL_14:
              *v17 = -16;
              --*(_DWORD *)(a1 + 1064);
              ++*(_DWORD *)(a1 + 1068);
            }
            else
            {
              v57 = 1;
              while ( v18 != -8 )
              {
                v58 = v57 + 1;
                v16 = v14 & (v57 + v16);
                v17 = (__int64 *)(v15 + 8LL * v16);
                v18 = *v17;
                if ( v13 == *v17 )
                  goto LABEL_14;
                v57 = v58;
              }
            }
          }
          v4 += 24;
          goto LABEL_16;
        }
      }
    }
    goto LABEL_3;
  }
  v21 = *(_DWORD *)(a1 + 1072);
  if ( !v21 )
    goto LABEL_16;
  v22 = v21 - 1;
  v23 = *(_QWORD *)(a1 + 1056);
  v24 = (v21 - 1) & (((unsigned int)*(_QWORD *)v4 >> 9) ^ ((unsigned int)*(_QWORD *)v4 >> 4));
  v25 = (__int64 *)(v23 + 8LL * v24);
  v26 = *v25;
  if ( *(_QWORD *)v4 != *v25 )
  {
    v51 = 1;
    while ( v26 != -8 )
    {
      v52 = v51 + 1;
      v24 = v22 & (v51 + v24);
      v25 = (__int64 *)(v23 + 8LL * v24);
      v26 = *v25;
      if ( *(_QWORD *)v4 == *v25 )
        goto LABEL_26;
      v51 = v52;
    }
LABEL_16:
    result = (__int64)v77;
    if ( v77 != v4 )
      goto LABEL_17;
    goto LABEL_27;
  }
LABEL_26:
  result = (__int64)v77;
  *v25 = -16;
  --*(_DWORD *)(a1 + 1064);
  ++*(_DWORD *)(a1 + 1068);
  if ( v77 == v4 )
    goto LABEL_27;
LABEL_17:
  v19 = v4 + 8;
  if ( (char *)result != v4 + 8 )
  {
    do
    {
      v20 = *(_QWORD *)v19;
      result = *(_QWORD *)(*(_QWORD *)a2 + 1056LL);
      if ( (__int64 (*)())result != sub_168DCA0 )
      {
        result = ((__int64 (__fastcall *)(__int64, __int64))result)(a2, v20);
        if ( !(_BYTE)result )
        {
          result = *(unsigned int *)(a1 + 1072);
          if ( (_DWORD)result )
          {
            v41 = result - 1;
            v42 = *(_QWORD *)(a1 + 1056);
            result = ((_DWORD)result - 1) & (((unsigned int)*(_QWORD *)v19 >> 9) ^ ((unsigned int)*(_QWORD *)v19 >> 4));
            v43 = (__int64 *)(v42 + 8 * result);
            v44 = *v43;
            if ( *(_QWORD *)v19 == *v43 )
            {
LABEL_40:
              *v43 = -16;
              --*(_DWORD *)(a1 + 1064);
              ++*(_DWORD *)(a1 + 1068);
            }
            else
            {
              v59 = 1;
              while ( v44 != -8 )
              {
                v60 = v59 + 1;
                result = v41 & (unsigned int)(v59 + result);
                v43 = (__int64 *)(v42 + 8LL * (unsigned int)result);
                v44 = *v43;
                if ( *(_QWORD *)v19 == *v43 )
                  goto LABEL_40;
                v59 = v60;
              }
            }
          }
          goto LABEL_20;
        }
        v20 = *(_QWORD *)v19;
      }
      *(_QWORD *)v4 = v20;
      v4 += 8;
LABEL_20:
      v19 += 8;
    }
    while ( v77 != v19 );
  }
LABEL_27:
  if ( *(char **)(a1 + 1088) != v4 )
    *(_QWORD *)(a1 + 1088) = v4;
  return result;
}
