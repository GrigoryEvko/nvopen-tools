// Function: sub_2608EC0
// Address: 0x2608ec0
//
__int64 __fastcall sub_2608EC0(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, char *a6, __int64 a7)
{
  __int64 result; // rax
  char *v8; // r15
  char *v9; // r14
  char *v10; // r13
  char *v11; // r12
  __int64 v12; // rbx
  char *v13; // r15
  __int64 v14; // r13
  __int64 v15; // r14
  char *v16; // r11
  __int64 v17; // r10
  __int64 v18; // rcx
  char *v19; // rax
  size_t v20; // r8
  size_t v21; // r8
  char *v22; // rax
  __int64 v23; // rdx
  signed __int64 v24; // rbx
  char *v25; // rcx
  unsigned int v26; // edx
  char *v27; // rsi
  unsigned int *v28; // r12
  unsigned int v29; // edx
  char *v30; // rax
  signed __int64 v31; // r8
  char *v32; // r8
  unsigned int v33; // eax
  size_t na; // [rsp+8h] [rbp-68h]
  size_t n; // [rsp+8h] [rbp-68h]
  int nb; // [rsp+8h] [rbp-68h]
  int v37; // [rsp+10h] [rbp-60h]
  int v38; // [rsp+10h] [rbp-60h]
  signed __int64 v39; // [rsp+10h] [rbp-60h]
  char *v40; // [rsp+10h] [rbp-60h]
  int v41; // [rsp+10h] [rbp-60h]
  int v42; // [rsp+10h] [rbp-60h]
  char *src; // [rsp+20h] [rbp-50h]
  char *srca; // [rsp+20h] [rbp-50h]
  char *srcb; // [rsp+20h] [rbp-50h]
  int srcc; // [rsp+20h] [rbp-50h]
  int srcd; // [rsp+20h] [rbp-50h]
  int srce; // [rsp+20h] [rbp-50h]
  char *v50; // [rsp+28h] [rbp-48h]
  char *v52; // [rsp+38h] [rbp-38h]

  result = a5;
  v8 = (char *)a3;
  v9 = a1;
  v10 = a6;
  v11 = a2;
  if ( a7 <= a5 )
    result = a7;
  if ( result >= a4 )
  {
LABEL_20:
    v24 = v11 - v9;
    if ( v11 != v9 )
      result = (__int64)memmove(v10, v9, v11 - v9);
    v25 = &v10[v24];
    if ( v10 != &v10[v24] )
    {
      while ( v8 != v11 )
      {
        result = *(unsigned int *)v11;
        v26 = *(_DWORD *)v10;
        if ( (unsigned int)result < *(_DWORD *)v10 )
        {
          *(_DWORD *)v9 = result;
          v11 += 4;
          v9 += 4;
          if ( v25 == v10 )
            return result;
        }
        else
        {
          result = v26;
          v10 += 4;
          v9 += 4;
          *((_DWORD *)v9 - 1) = v26;
          if ( v25 == v10 )
            return result;
        }
      }
    }
    if ( v25 != v10 )
      return (__int64)memmove(v9, v10, v25 - v10);
  }
  else
  {
    v12 = a5;
    if ( a7 < a5 )
    {
      v50 = a1;
      v13 = a2;
      v14 = a4;
      while ( 1 )
      {
        if ( v14 > v12 )
        {
          v30 = (char *)sub_25F6750(v13, a3, &v50[4 * (v14 / 2)]);
          v18 = v14 / 2;
          v52 = v30;
          v15 = (v30 - v13) >> 2;
        }
        else
        {
          v15 = v12 / 2;
          v52 = &v13[4 * (v12 / 2)];
          v16 = (char *)sub_25F6700(v50, (__int64)v13, v52);
          v18 = (__int64)&v16[-v17] >> 2;
        }
        v14 -= v18;
        if ( v14 <= v15 || a7 < v15 )
        {
          if ( a7 < v14 )
          {
            v42 = v18;
            srce = (int)v16;
            v19 = sub_25FA040(v16, v13, v52);
            LODWORD(v18) = v42;
            LODWORD(v16) = srce;
          }
          else
          {
            v19 = v52;
            if ( v14 )
            {
              v31 = v13 - v16;
              if ( v16 != v13 )
              {
                nb = v18;
                v39 = v13 - v16;
                srcb = v16;
                memmove(a6, v16, v13 - v16);
                LODWORD(v18) = nb;
                v31 = v39;
                v16 = srcb;
              }
              v32 = &a6[v31];
              if ( v52 != v13 )
              {
                v40 = v32;
                srcc = v18;
                v33 = (unsigned int)memmove(v16, v13, v52 - v13);
                v32 = v40;
                LODWORD(v18) = srcc;
                LODWORD(v16) = v33;
              }
              v41 = (int)v16;
              srcd = v18;
              v19 = (char *)sub_2608E90(a6, v32, (__int64)v52);
              LODWORD(v18) = srcd;
              LODWORD(v16) = v41;
            }
          }
        }
        else
        {
          v19 = v16;
          if ( v15 )
          {
            v20 = v52 - v13;
            if ( v52 != v13 )
            {
              na = (size_t)v16;
              v37 = v18;
              memmove(a6, v13, v52 - v13);
              v16 = (char *)na;
              LODWORD(v18) = v37;
              v20 = v52 - v13;
            }
            n = v20;
            v38 = v18;
            src = v16;
            sub_2608E90(v16, v13, (__int64)v52);
            v21 = n;
            v16 = src;
            LODWORD(v18) = v38;
            if ( n )
            {
              v22 = (char *)memmove(src, a6, n);
              LODWORD(v18) = v38;
              v21 = n;
              v16 = v22;
            }
            v19 = &v16[v21];
          }
        }
        v12 -= v15;
        srca = v19;
        sub_2608EC0((_DWORD)v50, (_DWORD)v16, (_DWORD)v19, v18, v15, (_DWORD)a6, a7);
        v23 = v12;
        result = (__int64)srca;
        if ( a7 <= v12 )
          v23 = a7;
        if ( v23 >= v14 )
        {
          v8 = (char *)a3;
          v10 = a6;
          v9 = srca;
          v11 = v52;
          goto LABEL_20;
        }
        if ( a7 >= v12 )
          break;
        v50 = srca;
        v13 = v52;
      }
      v8 = (char *)a3;
      v10 = a6;
      v9 = srca;
      v11 = v52;
    }
    if ( v8 != v11 )
      result = (__int64)memmove(v10, v11, v8 - v11);
    v27 = &v10[v8 - v11];
    if ( v9 == v11 )
      return (__int64)sub_2608E90(v10, v27, (__int64)v8);
    if ( v10 != v27 )
    {
      v28 = (unsigned int *)(v11 - 4);
      while ( 1 )
      {
        v29 = *v28;
        result = *((unsigned int *)v27 - 1);
        v27 -= 4;
        v8 -= 4;
        if ( (unsigned int)result < *v28 )
          break;
LABEL_39:
        *(_DWORD *)v8 = result;
        if ( v10 == v27 )
          return result;
      }
      while ( 1 )
      {
        *(_DWORD *)v8 = v29;
        if ( v28 == (unsigned int *)v9 )
          break;
        result = *(unsigned int *)v27;
        v29 = *--v28;
        v8 -= 4;
        if ( (unsigned int)result >= v29 )
          goto LABEL_39;
      }
      v27 += 4;
      return (__int64)sub_2608E90(v10, v27, (__int64)v8);
    }
  }
  return result;
}
