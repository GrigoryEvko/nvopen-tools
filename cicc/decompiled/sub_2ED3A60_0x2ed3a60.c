// Function: sub_2ED3A60
// Address: 0x2ed3a60
//
void *__fastcall sub_2ED3A60(
        __int64 *a1,
        __int64 *a2,
        char *a3,
        __int64 a4,
        __int64 a5,
        __int64 *a6,
        __int64 a7,
        __int128 a8)
{
  __int64 *v8; // r15
  __int64 *v9; // r14
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v12; // r13
  __int64 *v13; // r9
  __int64 v14; // r12
  __int64 *v15; // rax
  char *v16; // r9
  char *v17; // r10
  __int64 v18; // r11
  size_t v19; // rcx
  char *v20; // rax
  __int64 v21; // rax
  __int64 *v22; // rbx
  __int64 *v23; // r12
  __int64 v24; // rax
  __int64 *v25; // rdx
  __int64 v26; // rax
  __int64 v27; // r9
  void *result; // rax
  __int64 *v29; // rdi
  __int64 v30; // r13
  __int64 v31; // r14
  unsigned int v32; // r14d
  __int64 *v33; // r12
  __int64 *v34; // rbx
  __int64 *v35; // r12
  __int64 *v36; // r13
  __int64 *v37; // rdi
  __int64 v38; // r15
  __int64 v39; // r14
  __int64 v40; // rax
  __int64 *v41; // rdx
  __int64 v42; // rax
  __int64 v43; // r9
  bool v44; // al
  unsigned int v45; // r14d
  __int64 *v46; // rsi
  __int64 *v47; // rdi
  size_t v48; // rdx
  __int64 *v49; // rax
  size_t v50; // r8
  unsigned int v51; // eax
  char *v52; // rax
  char *v53; // [rsp+8h] [rbp-88h]
  char *v54; // [rsp+8h] [rbp-88h]
  int v55; // [rsp+10h] [rbp-80h]
  size_t v56; // [rsp+10h] [rbp-80h]
  int v57; // [rsp+10h] [rbp-80h]
  int v58; // [rsp+10h] [rbp-80h]
  int v59; // [rsp+10h] [rbp-80h]
  size_t v60; // [rsp+18h] [rbp-78h]
  int v61; // [rsp+18h] [rbp-78h]
  int v62; // [rsp+18h] [rbp-78h]
  size_t v63; // [rsp+18h] [rbp-78h]
  size_t v64; // [rsp+18h] [rbp-78h]
  int v65; // [rsp+18h] [rbp-78h]
  int v66; // [rsp+18h] [rbp-78h]
  __int64 *v67; // [rsp+20h] [rbp-70h]
  _BYTE src[24]; // [rsp+28h] [rbp-68h]
  char *srca; // [rsp+28h] [rbp-68h]
  char *srcb; // [rsp+28h] [rbp-68h]
  void *srcc; // [rsp+28h] [rbp-68h]
  char *srcd; // [rsp+28h] [rbp-68h]
  int srce; // [rsp+28h] [rbp-68h]
  __int64 *v74; // [rsp+30h] [rbp-60h]
  unsigned __int64 desta; // [rsp+40h] [rbp-50h]

  v8 = a2;
  *(__m128i *)&src[8] = _mm_loadu_si128((const __m128i *)&a8);
  v9 = *(__int64 **)&src[8];
  v10 = a5;
  if ( a7 <= a5 )
    v10 = a7;
  if ( a4 <= v10 )
    goto LABEL_22;
  v11 = a5;
  if ( a7 >= a5 )
    goto LABEL_42;
  v12 = a4;
  v13 = a2;
  while ( 1 )
  {
    *(_QWORD *)src = v13;
    if ( v12 > v11 )
    {
      v49 = sub_2ED2B10(v13, (__int64)a3, &a1[v12 / 2], *(__int64 **)&src[8], *(_QWORD **)&src[16]);
      v16 = *(char **)src;
      v17 = (char *)&a1[v12 / 2];
      v67 = v49;
      v18 = v12 / 2;
      v14 = ((__int64)v49 - *(_QWORD *)src) >> 3;
    }
    else
    {
      v14 = v11 / 2;
      v67 = &v13[v11 / 2];
      v15 = sub_2ED2C50(a1, (__int64)v13, v67, *(__int64 **)&src[8], *(_QWORD **)&src[16]);
      v16 = *(char **)src;
      v17 = (char *)v15;
      v18 = v15 - a1;
    }
    v12 -= v18;
    if ( v12 <= v14 || v14 > a7 )
    {
      if ( v12 > a7 )
      {
        v59 = v18;
        v66 = (int)v17;
        v52 = sub_2ED2490(v17, v16, (char *)v67);
        LODWORD(v18) = v59;
        LODWORD(v17) = v66;
        *(_QWORD *)src = v52;
      }
      else
      {
        *(_QWORD *)src = v67;
        if ( v12 )
        {
          v50 = v16 - v17;
          if ( v16 != v17 )
          {
            v57 = v18;
            v54 = v16;
            v63 = v16 - v17;
            srcd = v17;
            memmove(a6, v17, v16 - v17);
            v16 = v54;
            LODWORD(v18) = v57;
            v50 = v63;
            v17 = srcd;
          }
          if ( v16 != (char *)v67 )
          {
            v64 = v50;
            srce = v18;
            v51 = (unsigned int)memmove(v17, v16, (char *)v67 - v16);
            v50 = v64;
            LODWORD(v18) = srce;
            LODWORD(v17) = v51;
          }
          *(_QWORD *)src = (char *)v67 - v50;
          if ( v50 )
          {
            v58 = (int)v17;
            v65 = v18;
            memmove((char *)v67 - v50, a6, v50);
            LODWORD(v18) = v65;
            LODWORD(v17) = v58;
          }
        }
      }
    }
    else
    {
      *(_QWORD *)src = v17;
      if ( v14 )
      {
        v19 = (char *)v67 - v16;
        if ( v16 != (char *)v67 )
        {
          v55 = v18;
          v53 = v17;
          v60 = (char *)v67 - v16;
          srca = v16;
          memmove(a6, v16, (char *)v67 - v16);
          v17 = v53;
          LODWORD(v18) = v55;
          v19 = v60;
          v16 = srca;
        }
        if ( v16 != v17 )
        {
          v56 = v19;
          v61 = v18;
          srcb = v17;
          memmove((char *)v67 - (v16 - v17), v17, v16 - v17);
          v19 = v56;
          LODWORD(v18) = v61;
          v17 = srcb;
        }
        if ( v19 )
        {
          v62 = v18;
          srcc = (void *)v19;
          v20 = (char *)memmove(v17, a6, v19);
          LODWORD(v18) = v62;
          v19 = (size_t)srcc;
          v17 = v20;
        }
        *(_QWORD *)src = &v17[v19];
      }
    }
    v11 -= v14;
    sub_2ED3A60((_DWORD)a1, (_DWORD)v17, *(_DWORD *)src, v18, v14, (_DWORD)a6, a7, *(__int128 *)&src[8]);
    v21 = a7;
    if ( v11 <= a7 )
      v21 = v11;
    if ( v12 <= v21 )
    {
      v8 = v67;
      a1 = *(__int64 **)src;
LABEL_22:
      if ( v8 != a1 )
        memmove(a6, a1, (char *)v8 - (char *)a1);
      v22 = a6;
      v74 = (__int64 *)((char *)a6 + (char *)v8 - (char *)a1);
      if ( a6 == v74 || a3 == (char *)v8 )
      {
LABEL_61:
        result = v74;
        v46 = a6;
        if ( v74 != a6 )
        {
          v47 = a1;
          v48 = (char *)v74 - (char *)a6;
          return memmove(v47, v46, v48);
        }
        return result;
      }
      v23 = v9;
      while ( 1 )
      {
        v29 = (__int64 *)v23[8];
        v30 = *v22;
        v31 = *v8;
        if ( v29 )
        {
          v24 = sub_2E39EA0(v29, *v8);
          v25 = (__int64 *)v23[8];
          desta = v24;
          if ( v25 )
          {
            v26 = sub_2E39EA0(v25, v30);
            v25 = (__int64 *)v23[8];
            v27 = v26;
          }
          else
          {
            v27 = 0;
          }
          *(_QWORD *)src = v27;
          if ( !(unsigned __int8)sub_2EE68A0(**(_QWORD **)&src[16], v23[7], v25, 2) && *(_QWORD *)src | desta )
          {
            if ( desta >= *(_QWORD *)src )
              goto LABEL_38;
            goto LABEL_32;
          }
        }
        else
        {
          sub_2EE68A0(**(_QWORD **)&src[16], v23[7], 0, 2);
        }
        v32 = sub_2E5E7B0(v23[6], v31);
        if ( v32 >= (unsigned int)sub_2E5E7B0(v23[6], v30) )
        {
LABEL_38:
          result = (void *)*v22++;
          goto LABEL_33;
        }
LABEL_32:
        result = (void *)*v8++;
LABEL_33:
        *a1++ = (__int64)result;
        if ( v74 == v22 )
          return result;
        if ( a3 == (char *)v8 )
        {
          a6 = v22;
          goto LABEL_61;
        }
      }
    }
    if ( v11 <= a7 )
      break;
    v13 = v67;
    a1 = *(__int64 **)src;
  }
  v8 = v67;
  a1 = *(__int64 **)src;
LABEL_42:
  if ( a3 != (char *)v8 )
    memmove(a6, v8, a3 - (char *)v8);
  result = a6;
  v33 = (__int64 *)((char *)a6 + a3 - (char *)v8);
  if ( v8 == a1 )
  {
    v46 = a6;
    if ( a6 != v33 )
    {
      v48 = a3 - (char *)v8;
      v47 = v8;
      return memmove(v47, v46, v48);
    }
  }
  else if ( a6 != v33 )
  {
    v34 = v8 - 1;
    v35 = v33 - 1;
    v36 = *(__int64 **)&src[8];
LABEL_47:
    v37 = (__int64 *)v36[8];
    v38 = *v34;
    v39 = *v35;
    if ( v37 )
    {
LABEL_48:
      v40 = sub_2E39EA0(v37, v39);
      v41 = (__int64 *)v36[8];
      *(_QWORD *)&src[8] = v40;
      if ( v41 )
      {
        v42 = sub_2E39EA0(v41, v38);
        v41 = (__int64 *)v36[8];
        v43 = v42;
      }
      else
      {
        v43 = 0;
      }
      *(_QWORD *)src = v43;
      if ( !(unsigned __int8)sub_2EE68A0(**(_QWORD **)&src[16], v36[7], v41, 2) && *(_OWORD *)src != 0 )
      {
        v44 = *(_QWORD *)&src[8] < *(_QWORD *)src;
        goto LABEL_53;
      }
      goto LABEL_57;
    }
    while ( 1 )
    {
      sub_2EE68A0(**(_QWORD **)&src[16], v36[7], 0, 2);
LABEL_57:
      v45 = sub_2E5E7B0(v36[6], v39);
      v44 = v45 < (unsigned int)sub_2E5E7B0(v36[6], v38);
LABEL_53:
      a3 -= 8;
      if ( !v44 )
      {
        result = (void *)*v35;
        *(_QWORD *)a3 = *v35;
        if ( a6 == v35 )
          return result;
        --v35;
        goto LABEL_47;
      }
      result = (void *)*v34;
      *(_QWORD *)a3 = *v34;
      if ( v34 == a1 )
        break;
      v37 = (__int64 *)v36[8];
      --v34;
      v39 = *v35;
      v38 = *v34;
      if ( v37 )
        goto LABEL_48;
    }
    v46 = a6;
    if ( a6 != v35 + 1 )
    {
      v48 = (char *)(v35 + 1) - (char *)a6;
      v47 = (__int64 *)&a3[-v48];
      return memmove(v47, v46, v48);
    }
  }
  return result;
}
