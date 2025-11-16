// Function: sub_28EBBC0
// Address: 0x28ebbc0
//
char *__fastcall sub_28EBBC0(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, char *a6, char *a7)
{
  char *result; // rax
  char *v8; // r14
  char *v10; // r12
  char *v11; // rbx
  __int64 v12; // r15
  __int64 v13; // r9
  __int64 v14; // rbx
  __int64 v15; // r11
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // r11
  char *v19; // r9
  char *v20; // r12
  __int64 v21; // rcx
  char *v22; // r10
  size_t v23; // r10
  char *v24; // rcx
  _BYTE *v25; // rdx
  char *v26; // r12
  const void *v27; // rsi
  char *v28; // rdi
  size_t v29; // rdx
  __int64 v30; // rax
  size_t v31; // r8
  char *v32; // rax
  char *v33; // rax
  int v34; // [rsp+8h] [rbp-68h]
  char *v35; // [rsp+8h] [rbp-68h]
  int v36; // [rsp+10h] [rbp-60h]
  int v37; // [rsp+10h] [rbp-60h]
  int v38; // [rsp+10h] [rbp-60h]
  int v39; // [rsp+10h] [rbp-60h]
  int v40; // [rsp+10h] [rbp-60h]
  size_t v41; // [rsp+18h] [rbp-58h]
  size_t v42; // [rsp+18h] [rbp-58h]
  int v43; // [rsp+18h] [rbp-58h]
  int v44; // [rsp+18h] [rbp-58h]
  size_t v45; // [rsp+18h] [rbp-58h]
  int v46; // [rsp+18h] [rbp-58h]
  int v47; // [rsp+18h] [rbp-58h]
  char *src; // [rsp+28h] [rbp-48h]
  char *srcb; // [rsp+28h] [rbp-48h]
  int srcc; // [rsp+28h] [rbp-48h]
  void *srcd; // [rsp+28h] [rbp-48h]
  char *srca; // [rsp+28h] [rbp-48h]
  void *srce; // [rsp+28h] [rbp-48h]
  int srcf; // [rsp+28h] [rbp-48h]
  int srcg; // [rsp+28h] [rbp-48h]
  int srch; // [rsp+28h] [rbp-48h]
  char *dest; // [rsp+30h] [rbp-40h]
  char *v59; // [rsp+38h] [rbp-38h]

  result = (char *)a5;
  v8 = a1;
  v10 = a2;
  v11 = (char *)a3;
  if ( (__int64)a7 <= a5 )
    result = a7;
  if ( a4 <= (__int64)result )
  {
LABEL_22:
    if ( v10 != v8 )
      result = (char *)memmove(a6, v8, v10 - v8);
    v24 = &a6[v10 - v8];
    if ( v11 != v10 && a6 != v24 )
    {
      do
      {
        v25 = *(_BYTE **)a6;
        result = *(char **)v10;
        if ( *(_DWORD *)(*(_QWORD *)v10 + 32LL) < *(_DWORD *)(*(_QWORD *)a6 + 32LL) )
        {
          *(_QWORD *)v8 = result;
          v10 += 8;
          v8 += 8;
          if ( v24 == a6 )
            return result;
        }
        else
        {
          result = *(char **)a6;
          a6 += 8;
          v8 += 8;
          *((_QWORD *)v8 - 1) = v25;
          if ( v24 == a6 )
            return result;
        }
      }
      while ( v11 != v10 );
    }
    if ( v24 != a6 )
    {
      v27 = a6;
      v28 = v8;
      v29 = v24 - a6;
      return (char *)memmove(v28, v27, v29);
    }
  }
  else
  {
    v12 = a5;
    if ( (__int64)a7 < a5 )
    {
      v13 = (__int64)a2;
      v14 = a4;
      v15 = (__int64)a1;
      dest = a6;
      while ( 1 )
      {
        src = (char *)v13;
        if ( v12 < v14 )
        {
          v20 = (char *)(v15 + 8 * (v14 / 2));
          v30 = sub_28EA290(v13, a3, (__int64)v20);
          v19 = src;
          v21 = v14 / 2;
          v59 = (char *)v30;
          v16 = (v30 - (__int64)src) >> 3;
        }
        else
        {
          v16 = v12 / 2;
          v59 = (char *)(v13 + 8 * (v12 / 2));
          v17 = sub_28EA2E0(v15, v13, (__int64)v59);
          v19 = src;
          v20 = (char *)v17;
          v21 = (v17 - v18) >> 3;
        }
        v14 -= v21;
        if ( v14 <= v16 || (__int64)a7 < v16 )
        {
          if ( (__int64)a7 < v14 )
          {
            v47 = v18;
            srch = v21;
            v33 = sub_28EAA50(v20, v19, v59);
            LODWORD(v18) = v47;
            LODWORD(v21) = srch;
            v22 = v33;
          }
          else
          {
            v22 = v59;
            if ( v14 )
            {
              v31 = v19 - v20;
              if ( v19 != v20 )
              {
                v35 = v19;
                v39 = v18;
                v44 = v21;
                srce = (void *)(v19 - v20);
                memmove(dest, v20, v19 - v20);
                v19 = v35;
                LODWORD(v18) = v39;
                LODWORD(v21) = v44;
                v31 = (size_t)srce;
              }
              if ( v19 != v59 )
              {
                v40 = v18;
                v45 = v31;
                srcf = v21;
                memmove(v20, v19, v59 - v19);
                LODWORD(v18) = v40;
                v31 = v45;
                LODWORD(v21) = srcf;
              }
              v22 = &v59[-v31];
              if ( v31 )
              {
                v46 = v18;
                srcg = v21;
                v32 = (char *)memmove(&v59[-v31], dest, v31);
                LODWORD(v21) = srcg;
                LODWORD(v18) = v46;
                v22 = v32;
              }
            }
          }
        }
        else
        {
          v22 = v20;
          if ( v16 )
          {
            v23 = v59 - v19;
            if ( v19 != v59 )
            {
              v34 = v18;
              v36 = v21;
              v41 = v59 - v19;
              srcb = v19;
              memmove(dest, v19, v59 - v19);
              LODWORD(v18) = v34;
              LODWORD(v21) = v36;
              v23 = v41;
              v19 = srcb;
            }
            if ( v19 != v20 )
            {
              v37 = v18;
              v42 = v23;
              srcc = v21;
              memmove(&v59[-(v19 - v20)], v20, v19 - v20);
              LODWORD(v18) = v37;
              v23 = v42;
              LODWORD(v21) = srcc;
            }
            if ( v23 )
            {
              v38 = v18;
              v43 = v21;
              srcd = (void *)v23;
              memmove(v20, dest, v23);
              LODWORD(v18) = v38;
              LODWORD(v21) = v43;
              v23 = (size_t)srcd;
            }
            v22 = &v20[v23];
          }
        }
        v12 -= v16;
        srca = v22;
        sub_28EBBC0(v18, (_DWORD)v20, (_DWORD)v22, v21, v16, (_DWORD)dest, (__int64)a7);
        result = (char *)v12;
        if ( (__int64)a7 <= v12 )
          result = a7;
        if ( (__int64)result >= v14 )
        {
          v11 = (char *)a3;
          a6 = dest;
          v8 = srca;
          v10 = v59;
          goto LABEL_22;
        }
        if ( (__int64)a7 >= v12 )
          break;
        v13 = (__int64)v59;
        v15 = (__int64)srca;
      }
      v11 = (char *)a3;
      a6 = dest;
      v8 = srca;
      v10 = v59;
    }
    if ( v11 != v10 )
      memmove(a6, v10, v11 - v10);
    result = &a6[v11 - v10];
    if ( v8 == v10 )
    {
      if ( a6 != result )
      {
        v29 = v11 - v10;
        v28 = v10;
LABEL_59:
        v27 = a6;
        return (char *)memmove(v28, v27, v29);
      }
    }
    else if ( a6 != result )
    {
      v26 = v10 - 8;
LABEL_39:
      result -= 8;
      while ( 1 )
      {
        v11 -= 8;
        if ( *(_DWORD *)(*(_QWORD *)result + 32LL) >= *(_DWORD *)(*(_QWORD *)v26 + 32LL) )
        {
          *(_QWORD *)v11 = *(_QWORD *)result;
          if ( a6 != result )
            goto LABEL_39;
          return result;
        }
        *(_QWORD *)v11 = *(_QWORD *)v26;
        if ( v26 == v8 )
          break;
        v26 -= 8;
      }
      if ( a6 == result + 8 )
        return result;
      v29 = result + 8 - a6;
      v28 = &v11[-v29];
      goto LABEL_59;
    }
  }
  return result;
}
