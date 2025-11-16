// Function: sub_1F226E0
// Address: 0x1f226e0
//
__int64 __fastcall sub_1F226E0(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6, __int64 a7)
{
  __int64 result; // rax
  __int64 *v8; // r14
  __int64 *v10; // r12
  char *v11; // rbx
  __int64 v12; // r15
  __int64 *v13; // r9
  __int64 v14; // rbx
  __int64 *v15; // r11
  __int64 v16; // r13
  __int64 *v17; // rax
  __int64 v18; // r11
  char *v19; // r9
  __int64 *v20; // r12
  __int64 v21; // rcx
  char *v22; // r10
  size_t v23; // r10
  __int64 *v24; // r8
  __int64 v25; // rcx
  __int64 *v26; // r8
  __int64 *v27; // r12
  __int64 *v28; // rsi
  __int64 *v29; // rdi
  size_t v30; // rdx
  __int64 *v31; // rax
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
  char *src; // [rsp+28h] [rbp-48h]
  char *srcb; // [rsp+28h] [rbp-48h]
  int srcc; // [rsp+28h] [rbp-48h]
  void *srcd; // [rsp+28h] [rbp-48h]
  __int64 *srca; // [rsp+28h] [rbp-48h]
  void *srce; // [rsp+28h] [rbp-48h]
  int srcf; // [rsp+28h] [rbp-48h]
  int srcg; // [rsp+28h] [rbp-48h]
  int srch; // [rsp+28h] [rbp-48h]
  __int64 *dest; // [rsp+30h] [rbp-40h]
  __int64 *v60; // [rsp+38h] [rbp-38h]

  result = a5;
  v8 = a1;
  v10 = a2;
  v11 = (char *)a3;
  if ( a7 <= a5 )
    result = a7;
  if ( a4 <= result )
  {
LABEL_22:
    if ( v10 != v8 )
      result = (__int64)memmove(a6, v8, (char *)v10 - (char *)v8);
    v24 = (__int64 *)((char *)a6 + (char *)v10 - (char *)v8);
    if ( v11 != (char *)v10 && a6 != v24 )
    {
      do
      {
        v25 = *a6;
        result = *(_DWORD *)((*a6 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*a6 >> 1) & 3;
        if ( (*(_DWORD *)((*v10 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v10 >> 1) & 3) < (unsigned int)result )
        {
          *v8++ = *v10++;
          if ( v24 == a6 )
            return result;
        }
        else
        {
          ++a6;
          *v8++ = v25;
          if ( v24 == a6 )
            return result;
        }
      }
      while ( v11 != (char *)v10 );
    }
    if ( v24 != a6 )
    {
      v28 = a6;
      v29 = v8;
      v30 = (char *)v24 - (char *)a6;
      return (__int64)memmove(v29, v28, v30);
    }
  }
  else
  {
    v12 = a5;
    if ( a7 < a5 )
    {
      v13 = a2;
      v14 = a4;
      v15 = a1;
      dest = a6;
      while ( 1 )
      {
        src = (char *)v13;
        if ( v12 < v14 )
        {
          v20 = &v15[v14 / 2];
          v31 = sub_1F20EF0(v13, a3, v20);
          v19 = src;
          v21 = v14 / 2;
          v60 = v31;
          v16 = ((char *)v31 - src) >> 3;
        }
        else
        {
          v16 = v12 / 2;
          v60 = &v13[v12 / 2];
          v17 = sub_1F20E80(v15, (__int64)v13, v60);
          v19 = src;
          v20 = v17;
          v21 = ((__int64)v17 - v18) >> 3;
        }
        v14 -= v21;
        if ( v14 <= v16 || a7 < v16 )
        {
          if ( a7 < v14 )
          {
            v48 = v18;
            srch = v21;
            v34 = sub_1F20890((char *)v20, v19, (char *)v60);
            LODWORD(v18) = v48;
            LODWORD(v21) = srch;
            v22 = v34;
          }
          else
          {
            v22 = (char *)v60;
            if ( v14 )
            {
              v32 = v19 - (char *)v20;
              if ( v19 != (char *)v20 )
              {
                v36 = v19;
                v40 = v18;
                v45 = v21;
                srce = (void *)(v19 - (char *)v20);
                memmove(dest, v20, v19 - (char *)v20);
                v19 = v36;
                LODWORD(v18) = v40;
                LODWORD(v21) = v45;
                v32 = (size_t)srce;
              }
              if ( v19 != (char *)v60 )
              {
                v41 = v18;
                v46 = v32;
                srcf = v21;
                memmove(v20, v19, (char *)v60 - v19);
                LODWORD(v18) = v41;
                v32 = v46;
                LODWORD(v21) = srcf;
              }
              v22 = (char *)v60 - v32;
              if ( v32 )
              {
                v47 = v18;
                srcg = v21;
                v33 = (char *)memmove((char *)v60 - v32, dest, v32);
                LODWORD(v21) = srcg;
                LODWORD(v18) = v47;
                v22 = v33;
              }
            }
          }
        }
        else
        {
          v22 = (char *)v20;
          if ( v16 )
          {
            v23 = (char *)v60 - v19;
            if ( v19 != (char *)v60 )
            {
              v35 = v18;
              v37 = v21;
              v42 = (char *)v60 - v19;
              srcb = v19;
              memmove(dest, v19, (char *)v60 - v19);
              LODWORD(v18) = v35;
              LODWORD(v21) = v37;
              v23 = v42;
              v19 = srcb;
            }
            if ( v19 != (char *)v20 )
            {
              v38 = v18;
              v43 = v23;
              srcc = v21;
              memmove((char *)v60 - (v19 - (char *)v20), v20, v19 - (char *)v20);
              LODWORD(v18) = v38;
              v23 = v43;
              LODWORD(v21) = srcc;
            }
            if ( v23 )
            {
              v39 = v18;
              v44 = v21;
              srcd = (void *)v23;
              memmove(v20, dest, v23);
              LODWORD(v18) = v39;
              LODWORD(v21) = v44;
              v23 = (size_t)srcd;
            }
            v22 = (char *)v20 + v23;
          }
        }
        v12 -= v16;
        srca = (__int64 *)v22;
        sub_1F226E0(v18, (_DWORD)v20, (_DWORD)v22, v21, v16, (_DWORD)dest, a7);
        result = v12;
        if ( a7 <= v12 )
          result = a7;
        if ( result >= v14 )
        {
          v11 = (char *)a3;
          a6 = dest;
          v8 = srca;
          v10 = v60;
          goto LABEL_22;
        }
        if ( a7 >= v12 )
          break;
        v13 = v60;
        v15 = srca;
      }
      v11 = (char *)a3;
      a6 = dest;
      v8 = srca;
      v10 = v60;
    }
    if ( v11 != (char *)v10 )
      result = (__int64)memmove(a6, v10, v11 - (char *)v10);
    v26 = (__int64 *)((char *)a6 + v11 - (char *)v10);
    if ( v8 == v10 )
    {
      if ( a6 != v26 )
      {
        v30 = v11 - (char *)v10;
        v29 = v10;
LABEL_59:
        v28 = a6;
        return (__int64)memmove(v29, v28, v30);
      }
    }
    else if ( a6 != v26 )
    {
      v27 = v10 - 1;
LABEL_39:
      --v26;
      while ( 1 )
      {
        v11 -= 8;
        result = *(_DWORD *)((*v27 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v27 >> 1) & 3;
        if ( (*(_DWORD *)((*v26 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v26 >> 1) & 3) >= (unsigned int)result )
        {
          *(_QWORD *)v11 = *v26;
          if ( a6 != v26 )
            goto LABEL_39;
          return result;
        }
        *(_QWORD *)v11 = *v27;
        if ( v27 == v8 )
          break;
        --v27;
      }
      if ( a6 == v26 + 1 )
        return result;
      v30 = (char *)(v26 + 1) - (char *)a6;
      v29 = (__int64 *)&v11[-v30];
      goto LABEL_59;
    }
  }
  return result;
}
