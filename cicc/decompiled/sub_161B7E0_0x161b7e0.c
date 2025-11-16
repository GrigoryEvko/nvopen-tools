// Function: sub_161B7E0
// Address: 0x161b7e0
//
char *__fastcall sub_161B7E0(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, char *a6, char *a7)
{
  char *result; // rax
  char *v8; // r14
  char *v10; // r12
  char *v11; // rbx
  __int64 v12; // r15
  char *v13; // r9
  __int64 v14; // rbx
  char *v15; // r11
  __int64 v16; // r13
  char *v17; // r9
  __int64 v18; // r11
  char *v19; // r12
  __int64 v20; // rcx
  char *v21; // r10
  size_t v22; // r10
  char *v23; // rcx
  char *v24; // rdx
  char *v25; // r12
  unsigned __int64 v26; // rcx
  unsigned __int64 v27; // rdx
  char *v28; // rsi
  char *v29; // rdi
  size_t v30; // rdx
  char *v31; // rax
  size_t v32; // r8
  char *v33; // rax
  char *v34; // rax
  int v35; // [rsp+8h] [rbp-68h]
  char *v36; // [rsp+8h] [rbp-68h]
  int v37; // [rsp+10h] [rbp-60h]
  int v38; // [rsp+10h] [rbp-60h]
  int v39; // [rsp+10h] [rbp-60h]
  int v40; // [rsp+10h] [rbp-60h]
  int v41; // [rsp+10h] [rbp-60h]
  size_t v42; // [rsp+18h] [rbp-58h]
  size_t v43; // [rsp+18h] [rbp-58h]
  int v44; // [rsp+18h] [rbp-58h]
  int v45; // [rsp+18h] [rbp-58h]
  size_t v46; // [rsp+18h] [rbp-58h]
  int v47; // [rsp+18h] [rbp-58h]
  int v48; // [rsp+18h] [rbp-58h]
  char *v50; // [rsp+28h] [rbp-48h]
  int v51; // [rsp+28h] [rbp-48h]
  size_t v52; // [rsp+28h] [rbp-48h]
  char *v53; // [rsp+28h] [rbp-48h]
  size_t v54; // [rsp+28h] [rbp-48h]
  int v55; // [rsp+28h] [rbp-48h]
  int v56; // [rsp+28h] [rbp-48h]
  int v57; // [rsp+28h] [rbp-48h]
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
    v23 = &a6[v10 - v8];
    if ( v11 != v10 && a6 != v23 )
    {
      do
      {
        v24 = *(char **)a6;
        result = *(char **)v10;
        if ( *(_QWORD *)a6 > *(_QWORD *)v10 )
        {
          *(_QWORD *)v8 = result;
          v10 += 8;
          v8 += 8;
          if ( v23 == a6 )
            return result;
        }
        else
        {
          result = *(char **)a6;
          a6 += 8;
          v8 += 8;
          *((_QWORD *)v8 - 1) = v24;
          if ( v23 == a6 )
            return result;
        }
      }
      while ( v11 != v10 );
    }
    if ( v23 != a6 )
    {
      v28 = a6;
      v29 = v8;
      v30 = v23 - a6;
      return (char *)memmove(v29, v28, v30);
    }
  }
  else
  {
    v12 = a5;
    if ( (__int64)a7 < a5 )
    {
      v13 = a2;
      v14 = a4;
      v15 = a1;
      dest = a6;
      while ( 1 )
      {
        if ( v12 < v14 )
        {
          v19 = &v15[8 * (v14 / 2)];
          v31 = (char *)sub_161B3C0(v13, a3, v19);
          v20 = v14 / 2;
          v59 = v31;
          v16 = (v31 - v17) >> 3;
        }
        else
        {
          v16 = v12 / 2;
          v59 = &v13[8 * (v12 / 2)];
          v19 = (char *)sub_161B410(v15, (__int64)v13, v59);
          v20 = (__int64)&v19[-v18] >> 3;
        }
        v14 -= v20;
        if ( v14 <= v16 || (__int64)a7 < v16 )
        {
          if ( (__int64)a7 < v14 )
          {
            v48 = v18;
            v57 = v20;
            v34 = sub_161B460(v19, v17, v59);
            LODWORD(v18) = v48;
            LODWORD(v20) = v57;
            v21 = v34;
          }
          else
          {
            v21 = v59;
            if ( v14 )
            {
              v32 = v17 - v19;
              if ( v17 != v19 )
              {
                v36 = v17;
                v40 = v18;
                v45 = v20;
                v54 = v17 - v19;
                memmove(dest, v19, v17 - v19);
                v17 = v36;
                LODWORD(v18) = v40;
                LODWORD(v20) = v45;
                v32 = v54;
              }
              if ( v17 != v59 )
              {
                v41 = v18;
                v46 = v32;
                v55 = v20;
                memmove(v19, v17, v59 - v17);
                LODWORD(v18) = v41;
                v32 = v46;
                LODWORD(v20) = v55;
              }
              v21 = &v59[-v32];
              if ( v32 )
              {
                v47 = v18;
                v56 = v20;
                v33 = (char *)memmove(&v59[-v32], dest, v32);
                LODWORD(v20) = v56;
                LODWORD(v18) = v47;
                v21 = v33;
              }
            }
          }
        }
        else
        {
          v21 = v19;
          if ( v16 )
          {
            v22 = v59 - v17;
            if ( v17 != v59 )
            {
              v35 = v18;
              v37 = v20;
              v42 = v59 - v17;
              v50 = v17;
              memmove(dest, v17, v59 - v17);
              LODWORD(v18) = v35;
              LODWORD(v20) = v37;
              v22 = v42;
              v17 = v50;
            }
            if ( v17 != v19 )
            {
              v38 = v18;
              v43 = v22;
              v51 = v20;
              memmove(&v59[-(v17 - v19)], v19, v17 - v19);
              LODWORD(v18) = v38;
              v22 = v43;
              LODWORD(v20) = v51;
            }
            if ( v22 )
            {
              v39 = v18;
              v44 = v20;
              v52 = v22;
              memmove(v19, dest, v22);
              LODWORD(v18) = v39;
              LODWORD(v20) = v44;
              v22 = v52;
            }
            v21 = &v19[v22];
          }
        }
        v12 -= v16;
        v53 = v21;
        sub_161B7E0(v18, (_DWORD)v19, (_DWORD)v21, v20, v16, (_DWORD)dest, (__int64)a7);
        result = (char *)v12;
        if ( (__int64)a7 <= v12 )
          result = a7;
        if ( (__int64)result >= v14 )
        {
          v11 = (char *)a3;
          a6 = dest;
          v8 = v53;
          v10 = v59;
          goto LABEL_22;
        }
        if ( (__int64)a7 >= v12 )
          break;
        v13 = v59;
        v15 = v53;
      }
      v11 = (char *)a3;
      a6 = dest;
      v8 = v53;
      v10 = v59;
    }
    if ( v11 != v10 )
      memmove(a6, v10, v11 - v10);
    result = &a6[v11 - v10];
    if ( v8 == v10 )
    {
      if ( a6 != result )
      {
        v30 = v11 - v10;
        v29 = v10;
        goto LABEL_58;
      }
    }
    else if ( a6 != result )
    {
      v25 = v10 - 8;
      while ( 1 )
      {
        v26 = *(_QWORD *)v25;
        v27 = *((_QWORD *)result - 1);
        result -= 8;
        v11 -= 8;
        if ( *(_QWORD *)v25 > v27 )
          break;
LABEL_42:
        *(_QWORD *)v11 = v27;
        if ( a6 == result )
          return result;
      }
      while ( 1 )
      {
        *(_QWORD *)v11 = v26;
        if ( v25 == v8 )
          break;
        v27 = *(_QWORD *)result;
        v26 = *((_QWORD *)v25 - 1);
        v25 -= 8;
        v11 -= 8;
        if ( v26 <= *(_QWORD *)result )
          goto LABEL_42;
      }
      if ( a6 != result + 8 )
      {
        v30 = result + 8 - a6;
        v29 = &v11[-v30];
LABEL_58:
        v28 = a6;
        return (char *)memmove(v29, v28, v30);
      }
    }
  }
  return result;
}
