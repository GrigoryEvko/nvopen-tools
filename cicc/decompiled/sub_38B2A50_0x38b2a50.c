// Function: sub_38B2A50
// Address: 0x38b2a50
//
__int64 __fastcall sub_38B2A50(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax
  unsigned int v5; // eax
  int v6; // eax
  _QWORD *v7; // rdi
  char *v8; // r9
  void *v9; // r8
  unsigned __int64 v10; // r13
  signed __int64 v11; // r10
  char *v12; // rsi
  char *v13; // rax
  char *v14; // rdx
  _QWORD *v15; // rdx
  signed __int64 v16; // rax
  char *v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  char *v20; // rsi
  _QWORD *v21; // rcx
  char **v22; // r11
  char *v23; // rax
  unsigned __int64 v24; // r9
  __int64 v25; // rax
  char *v26; // r8
  size_t v27; // rdx
  _QWORD *v28; // rax
  _QWORD *v29; // rdx
  char *v30; // r10
  char *v31; // r8
  signed __int64 v32; // r9
  _QWORD *v33; // rcx
  char v34; // di
  __int64 v35; // rax
  int v36; // eax
  unsigned __int64 v37; // rsi
  unsigned __int64 v38; // rsi
  _QWORD *v39; // rax
  signed __int64 v40; // rdi
  char **v41; // [rsp+0h] [rbp-D0h]
  _QWORD *v42; // [rsp+8h] [rbp-C8h]
  char *v43; // [rsp+8h] [rbp-C8h]
  unsigned __int64 v44; // [rsp+10h] [rbp-C0h]
  _QWORD *v45; // [rsp+10h] [rbp-C0h]
  char **v46; // [rsp+18h] [rbp-B8h]
  char *v47; // [rsp+18h] [rbp-B8h]
  unsigned __int64 v48; // [rsp+18h] [rbp-B8h]
  char *v49; // [rsp+20h] [rbp-B0h]
  char *v50; // [rsp+20h] [rbp-B0h]
  _QWORD *v51; // [rsp+28h] [rbp-A8h]
  _QWORD *v52; // [rsp+28h] [rbp-A8h]
  signed __int64 v53; // [rsp+30h] [rbp-A0h]
  _QWORD *v54; // [rsp+30h] [rbp-A0h]
  unsigned __int8 v55; // [rsp+38h] [rbp-98h]
  void *src; // [rsp+40h] [rbp-90h] BYREF
  char *v57; // [rsp+48h] [rbp-88h]
  __int64 v58; // [rsp+50h] [rbp-80h]
  __m128i v59; // [rsp+60h] [rbp-70h] BYREF
  _QWORD v60[2]; // [rsp+70h] [rbp-60h] BYREF
  _QWORD v61[2]; // [rsp+80h] [rbp-50h] BYREF
  char v62; // [rsp+90h] [rbp-40h]
  char v63; // [rsp+91h] [rbp-3Fh]

  if ( (unsigned __int8)sub_388AF10(a1, 360, "expected 'resByArg' here")
    || (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here") )
  {
    return 1;
  }
LABEL_6:
  src = 0;
  v57 = 0;
  v58 = 0;
  if ( (unsigned __int8)sub_388FA70(a1, (__int64)&src)
    || (unsigned __int8)sub_388AF10(a1, 4, "expected ',' here")
    || (unsigned __int8)sub_388AF10(a1, 361, "expected 'byArg here")
    || (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here")
    || (unsigned __int8)sub_388AF10(a1, 343, "expected 'kind' here")
    || (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here") )
  {
    goto LABEL_72;
  }
  v5 = *(_DWORD *)(a1 + 64);
  v59.m128i_i32[0] = 0;
  v59.m128i_i64[1] = 0;
  v60[0] = 0;
  if ( v5 == 363 )
  {
    v59.m128i_i32[0] = 2;
  }
  else
  {
    if ( v5 <= 0x16B )
    {
      if ( v5 == 356 )
        goto LABEL_18;
      if ( v5 == 362 )
      {
        v59.m128i_i32[0] = 1;
        goto LABEL_18;
      }
LABEL_75:
      v38 = *(_QWORD *)(a1 + 56);
      v63 = 1;
      v61[0] = "unexpected WholeProgramDevirtResolution::ByArg kind";
      v62 = 3;
      result = sub_38814C0(a1 + 8, v38, (__int64)v61);
      goto LABEL_63;
    }
    if ( v5 != 364 )
      goto LABEL_75;
    v59.m128i_i32[0] = 3;
  }
LABEL_18:
  v6 = sub_3887100(a1 + 8);
  *(_DWORD *)(a1 + 64) = v6;
  if ( v6 != 4 )
  {
LABEL_19:
    if ( !(unsigned __int8)sub_388AF10(a1, 13, "expected ')' here") )
    {
      v7 = (_QWORD *)a2[2];
      if ( !v7 )
      {
        v10 = (unsigned __int64)(a2 + 1);
        goto LABEL_39;
      }
      v8 = v57;
      v9 = src;
      v10 = (unsigned __int64)(a2 + 1);
      v11 = v57 - (_BYTE *)src;
      while ( 1 )
      {
        v12 = (char *)v7[5];
        v13 = (char *)v7[4];
        if ( v12 - v13 > v11 )
          v12 = &v13[v11];
        v14 = (char *)src;
        if ( v13 == v12 )
        {
LABEL_65:
          if ( v57 == v14 )
          {
LABEL_66:
            v10 = (unsigned __int64)v7;
            v7 = (_QWORD *)v7[2];
            goto LABEL_30;
          }
        }
        else
        {
          while ( *(_QWORD *)v13 >= *(_QWORD *)v14 )
          {
            if ( *(_QWORD *)v13 > *(_QWORD *)v14 )
              goto LABEL_66;
            v13 += 8;
            v14 += 8;
            if ( v12 == v13 )
              goto LABEL_65;
          }
        }
        v7 = (_QWORD *)v7[3];
LABEL_30:
        if ( !v7 )
        {
          if ( (_QWORD *)v10 == a2 + 1 )
            goto LABEL_39;
          v15 = *(_QWORD **)(v10 + 32);
          v16 = *(_QWORD *)(v10 + 40) - (_QWORD)v15;
          if ( v11 > v16 )
            v8 = (char *)src + v16;
          if ( src == v8 )
          {
LABEL_76:
            if ( *(_QWORD **)(v10 + 40) != v15 )
              goto LABEL_39;
            goto LABEL_49;
          }
          v17 = (char *)src;
          while ( *(_QWORD *)v17 >= *v15 )
          {
            if ( *(_QWORD *)v17 > *v15 )
              goto LABEL_49;
            v17 += 8;
            ++v15;
            if ( v8 == v17 )
              goto LABEL_76;
          }
LABEL_39:
          v51 = (_QWORD *)v10;
          v18 = sub_22077B0(0x50u);
          v20 = (char *)src;
          v21 = a2 + 1;
          v10 = v18;
          v22 = (char **)(v18 + 32);
          v23 = v57;
          *(_QWORD *)(v10 + 32) = 0;
          *(_QWORD *)(v10 + 40) = 0;
          v24 = v23 - v20;
          *(_QWORD *)(v10 + 48) = 0;
          if ( v23 == v20 )
          {
            v53 = 0;
            v27 = 0;
            v26 = 0;
          }
          else
          {
            v53 = v23 - v20;
            if ( v24 > 0x7FFFFFFFFFFFFFF8LL )
              sub_4261EA(80, v20, v19);
            v46 = v22;
            v25 = sub_22077B0(v24);
            v20 = (char *)src;
            v21 = a2 + 1;
            v26 = (char *)v25;
            v23 = v57;
            v22 = v46;
            v24 = v57 - (_BYTE *)src;
            v27 = v57 - (_BYTE *)src;
          }
          *(_QWORD *)(v10 + 32) = v26;
          *(_QWORD *)(v10 + 40) = v26;
          *(_QWORD *)(v10 + 48) = &v26[v53];
          if ( v20 == v23 )
          {
            *(_QWORD *)(v10 + 40) = &v26[v27];
            *(_DWORD *)(v10 + 56) = 0;
            *(_QWORD *)(v10 + 64) = 0;
            *(_QWORD *)(v10 + 72) = 0;
            v43 = v26;
            v45 = v21;
            v48 = v24;
            v50 = &v26[v27];
            v28 = sub_14F7820(a2, v51, v22);
            v30 = v50;
            v32 = v48;
            v33 = v45;
            v31 = v43;
            if ( v29 )
              goto LABEL_44;
            if ( v43 )
              goto LABEL_82;
          }
          else
          {
            v42 = v21;
            v44 = v24;
            v47 = v26;
            v41 = v22;
            v49 = &v26[v27];
            memmove(v26, v20, v27);
            *(_DWORD *)(v10 + 56) = 0;
            *(_QWORD *)(v10 + 40) = v49;
            *(_QWORD *)(v10 + 64) = 0;
            *(_QWORD *)(v10 + 72) = 0;
            v28 = sub_14F7820(a2, v51, v41);
            v30 = v49;
            v31 = v47;
            v32 = v44;
            v33 = v42;
            if ( v29 )
            {
LABEL_44:
              if ( v33 == v29 || v28 )
                goto LABEL_46;
              v39 = (_QWORD *)v29[4];
              v40 = v29[5] - (_QWORD)v39;
              if ( v32 > v40 )
                v30 = &v31[v40];
              if ( v31 == v30 )
              {
LABEL_91:
                v34 = v39 != (_QWORD *)v29[5];
              }
              else
              {
                while ( *(_QWORD *)v31 >= *v39 )
                {
                  if ( *(_QWORD *)v31 > *v39 )
                  {
                    v34 = 0;
                    goto LABEL_47;
                  }
                  v31 += 8;
                  ++v39;
                  if ( v30 == v31 )
                    goto LABEL_91;
                }
LABEL_46:
                v34 = 1;
              }
LABEL_47:
              sub_220F040(v34, v10, v29, v33);
              ++a2[5];
LABEL_48:
              v9 = src;
LABEL_49:
              v35 = v60[0];
              *(__m128i *)(v10 + 56) = _mm_loadu_si128(&v59);
              *(_QWORD *)(v10 + 72) = v35;
              if ( v9 )
                j_j___libc_free_0((unsigned __int64)v9);
              if ( *(_DWORD *)(a1 + 64) != 4 )
                return sub_388AF10(a1, 13, "expected ')' here");
              *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
              goto LABEL_6;
            }
LABEL_82:
            v52 = v28;
            j_j___libc_free_0((unsigned __int64)v31);
            v28 = v52;
          }
          v54 = v28;
          j_j___libc_free_0(v10);
          v10 = (unsigned __int64)v54;
          goto LABEL_48;
        }
      }
    }
LABEL_72:
    result = 1;
    goto LABEL_63;
  }
  while ( 1 )
  {
    v36 = sub_3887100(a1 + 8);
    *(_DWORD *)(a1 + 64) = v36;
    if ( v36 == 366 )
    {
      *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
      if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here") || (unsigned __int8)sub_388BA90(a1, v60) )
        goto LABEL_72;
      goto LABEL_56;
    }
    if ( v36 == 367 )
    {
      *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
      if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
        || (unsigned __int8)sub_388BA90(a1, (_DWORD *)v60 + 1) )
      {
        goto LABEL_72;
      }
      goto LABEL_56;
    }
    if ( v36 != 365 )
      break;
    *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
    if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
      || (unsigned __int8)sub_388BD80(a1, &v59.m128i_i64[1]) )
    {
      goto LABEL_72;
    }
LABEL_56:
    if ( *(_DWORD *)(a1 + 64) != 4 )
      goto LABEL_19;
  }
  v37 = *(_QWORD *)(a1 + 56);
  v63 = 1;
  v62 = 3;
  v61[0] = "expected optional whole program devirt field";
  result = sub_38814C0(a1 + 8, v37, (__int64)v61);
LABEL_63:
  if ( src )
  {
    v55 = result;
    j_j___libc_free_0((unsigned __int64)src);
    return v55;
  }
  return result;
}
