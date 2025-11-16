// Function: sub_F08760
// Address: 0xf08760
//
void **__fastcall sub_F08760(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, void **a6, __int64 a7)
{
  __int64 v7; // rax
  __int64 *v8; // r14
  __int64 *v10; // r12
  __int64 *v11; // rbx
  __int64 v12; // r15
  __int64 *v13; // r9
  __int64 v14; // rbx
  __int64 *v15; // r11
  __int64 v16; // r13
  __int64 *v17; // rax
  int v18; // r11d
  char *v19; // r9
  __int64 *v20; // r12
  __int64 v21; // rcx
  char *v22; // r10
  size_t v23; // r10
  __int64 v24; // rax
  signed __int64 v25; // r15
  void **result; // rax
  __int64 *v27; // rbx
  __int64 *v28; // r14
  __int64 v29; // r12
  __int64 v30; // r15
  __int64 v31; // rax
  __int64 *v32; // r14
  __int64 *i; // r12
  __int64 v34; // r13
  __int64 v35; // r15
  __int64 v36; // rax
  size_t v37; // rdx
  void **v38; // rsi
  __int64 *v39; // rdi
  __int64 *v40; // rax
  size_t v41; // r8
  char *v42; // rax
  char *v43; // rax
  int v44; // [rsp+8h] [rbp-68h]
  char *v45; // [rsp+8h] [rbp-68h]
  int v46; // [rsp+10h] [rbp-60h]
  int v47; // [rsp+10h] [rbp-60h]
  int v48; // [rsp+10h] [rbp-60h]
  int v49; // [rsp+10h] [rbp-60h]
  int v50; // [rsp+10h] [rbp-60h]
  int v51; // [rsp+10h] [rbp-60h]
  __int64 *src; // [rsp+18h] [rbp-58h]
  void *srca; // [rsp+18h] [rbp-58h]
  void *srcb; // [rsp+18h] [rbp-58h]
  int srcc; // [rsp+18h] [rbp-58h]
  int srcd; // [rsp+18h] [rbp-58h]
  void *srce; // [rsp+18h] [rbp-58h]
  int srcf; // [rsp+18h] [rbp-58h]
  int srcg; // [rsp+18h] [rbp-58h]
  __int64 *v61; // [rsp+28h] [rbp-48h]
  char *v62; // [rsp+28h] [rbp-48h]
  int v63; // [rsp+28h] [rbp-48h]
  void *v64; // [rsp+28h] [rbp-48h]
  __int64 *v65; // [rsp+28h] [rbp-48h]
  __int64 *v66; // [rsp+28h] [rbp-48h]
  void *v67; // [rsp+28h] [rbp-48h]
  int v68; // [rsp+28h] [rbp-48h]
  int v69; // [rsp+28h] [rbp-48h]
  int v70; // [rsp+28h] [rbp-48h]
  void **dest; // [rsp+30h] [rbp-40h]
  __int64 *desta; // [rsp+30h] [rbp-40h]
  __int64 *destb; // [rsp+30h] [rbp-40h]
  __int64 *v74; // [rsp+38h] [rbp-38h]
  char *v75; // [rsp+38h] [rbp-38h]
  __int64 *v76; // [rsp+38h] [rbp-38h]

  v7 = a5;
  v8 = a1;
  v10 = a2;
  v11 = (__int64 *)a3;
  if ( a7 <= a5 )
    v7 = a7;
  if ( a4 <= v7 )
  {
LABEL_22:
    v25 = (char *)v10 - (char *)v8;
    if ( v8 != v10 )
      memmove(a6, v8, (char *)v10 - (char *)v8);
    result = (void **)((char *)a6 + v25);
    v75 = (char *)a6 + v25;
    if ( (void **)((char *)a6 + v25) != a6 )
    {
      if ( v11 != v10 )
      {
        desta = v11;
        v27 = v8;
        v28 = v10;
        do
        {
          v29 = *v28;
          v30 = sub_B140A0((__int64)*a6);
          v31 = sub_B140A0(v29);
          if ( sub_B445A0(v30, v31) )
          {
            result = (void **)*v28;
            ++v27;
            ++v28;
            *(v27 - 1) = (__int64)result;
            if ( v75 == (char *)a6 )
              return result;
          }
          else
          {
            result = (void **)*a6;
            ++v27;
            ++a6;
            *(v27 - 1) = (__int64)result;
            if ( v75 == (char *)a6 )
              return result;
          }
        }
        while ( desta != v28 );
        v8 = v27;
      }
      if ( v75 != (char *)a6 )
      {
        v37 = v75 - (char *)a6;
        v38 = a6;
        v39 = v8;
        return (void **)memmove(v39, v38, v37);
      }
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
        if ( v12 < v14 )
        {
          v49 = (int)v15;
          v66 = v13;
          v20 = &v15[v14 / 2];
          v40 = sub_F07240(v13, a3, v20);
          v19 = (char *)v66;
          v21 = v14 / 2;
          v74 = v40;
          v18 = v49;
          v16 = v40 - v66;
        }
        else
        {
          src = v13;
          v61 = v15;
          v16 = v12 / 2;
          v74 = &v13[v12 / 2];
          v17 = sub_F072E0(v15, (__int64)v13, v74);
          v18 = (int)v61;
          v19 = (char *)src;
          v20 = v17;
          v21 = v17 - v61;
        }
        v14 -= v21;
        if ( v14 <= v16 || v16 > a7 )
        {
          if ( v14 > a7 )
          {
            srcg = v18;
            v70 = v21;
            v43 = sub_F07A80((char *)v20, v19, (char *)v74);
            v18 = srcg;
            LODWORD(v21) = v70;
            v22 = v43;
          }
          else
          {
            v22 = (char *)v74;
            if ( v14 )
            {
              v41 = v19 - (char *)v20;
              if ( v19 != (char *)v20 )
              {
                v45 = v19;
                v50 = v18;
                srcd = v21;
                v67 = (void *)(v19 - (char *)v20);
                memmove(dest, v20, v19 - (char *)v20);
                v19 = v45;
                v18 = v50;
                LODWORD(v21) = srcd;
                v41 = (size_t)v67;
              }
              if ( v19 != (char *)v74 )
              {
                v51 = v18;
                srce = (void *)v41;
                v68 = v21;
                memmove(v20, v19, (char *)v74 - v19);
                v18 = v51;
                v41 = (size_t)srce;
                LODWORD(v21) = v68;
              }
              v22 = (char *)v74 - v41;
              if ( v41 )
              {
                srcf = v18;
                v69 = v21;
                v42 = (char *)memmove((char *)v74 - v41, dest, v41);
                LODWORD(v21) = v69;
                v18 = srcf;
                v22 = v42;
              }
            }
          }
        }
        else
        {
          v22 = (char *)v20;
          if ( v16 )
          {
            v23 = (char *)v74 - v19;
            if ( v19 != (char *)v74 )
            {
              v44 = v18;
              v46 = v21;
              srca = (void *)((char *)v74 - v19);
              v62 = v19;
              memmove(dest, v19, (char *)v74 - v19);
              v18 = v44;
              LODWORD(v21) = v46;
              v23 = (size_t)srca;
              v19 = v62;
            }
            if ( v19 != (char *)v20 )
            {
              v47 = v18;
              srcb = (void *)v23;
              v63 = v21;
              memmove((char *)v74 - (v19 - (char *)v20), v20, v19 - (char *)v20);
              v18 = v47;
              v23 = (size_t)srcb;
              LODWORD(v21) = v63;
            }
            if ( v23 )
            {
              v48 = v18;
              srcc = v21;
              v64 = (void *)v23;
              memmove(v20, dest, v23);
              v18 = v48;
              LODWORD(v21) = srcc;
              v23 = (size_t)v64;
            }
            v22 = (char *)v20 + v23;
          }
        }
        v12 -= v16;
        v65 = (__int64 *)v22;
        sub_F08760(v18, (_DWORD)v20, (_DWORD)v22, v21, v16, (_DWORD)dest, a7);
        v24 = a7;
        if ( v12 <= a7 )
          v24 = v12;
        if ( v14 <= v24 )
        {
          v11 = (__int64 *)a3;
          a6 = dest;
          v8 = v65;
          v10 = v74;
          goto LABEL_22;
        }
        if ( v12 <= a7 )
          break;
        v13 = v74;
        v15 = v65;
      }
      v11 = (__int64 *)a3;
      a6 = dest;
      v8 = v65;
      v10 = v74;
    }
    if ( v11 != v10 )
      memmove(a6, v10, (char *)v11 - (char *)v10);
    result = (void **)((char *)a6 + (char *)v11 - (char *)v10);
    if ( v8 == v10 )
    {
      if ( a6 != result )
      {
        v37 = (char *)v11 - (char *)v10;
        v39 = v10;
        goto LABEL_59;
      }
    }
    else if ( a6 != result )
    {
      v76 = (__int64 *)a6;
      destb = v8;
      v32 = (__int64 *)(result - 1);
      for ( i = v10 - 1; ; --i )
      {
        while ( 1 )
        {
          v34 = *v32;
          --v11;
          v35 = sub_B140A0(*i);
          v36 = sub_B140A0(v34);
          if ( sub_B445A0(v35, v36) )
            break;
          result = (void **)*v32;
          *v11 = *v32;
          if ( v76 == v32 )
            return result;
          --v32;
        }
        result = (void **)*i;
        *v11 = *i;
        if ( i == destb )
          break;
      }
      a6 = (void **)v76;
      if ( v76 != v32 + 1 )
      {
        v37 = (char *)(v32 + 1) - (char *)v76;
        v39 = (__int64 *)((char *)v11 - v37);
LABEL_59:
        v38 = a6;
        return (void **)memmove(v39, v38, v37);
      }
    }
  }
  return result;
}
