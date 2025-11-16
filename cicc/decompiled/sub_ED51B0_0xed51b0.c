// Function: sub_ED51B0
// Address: 0xed51b0
//
__int64 __fastcall sub_ED51B0(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        unsigned __int64 *a4,
        unsigned __int64 a5,
        __int64 a6)
{
  unsigned __int64 *v9; // rbx
  unsigned __int64 *v10; // r12
  __int64 v11; // r8
  char *v12; // r9
  char *v13; // rdx
  __int64 v14; // rax
  char *v15; // rcx
  unsigned __int64 v16; // r11
  __int64 v17; // rsi
  void *v18; // r10
  signed __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdi
  bool v22; // cf
  unsigned __int64 v23; // rax
  __int64 *v24; // rax
  __int64 v25; // r11
  __int64 v26; // rdi
  __int64 result; // rax
  __int64 v28; // rdx
  _QWORD *v29; // r12
  __int64 v30; // r13
  char *v31; // rbx
  unsigned __int64 v32; // rbx
  char *v33; // rcx
  void *v34; // r14
  __int64 v35; // rbx
  char *v36; // rax
  __int64 v37; // r8
  __int64 v38; // rax
  signed __int64 v39; // [rsp+8h] [rbp-78h]
  __int64 v40; // [rsp+10h] [rbp-70h]
  void *v41; // [rsp+10h] [rbp-70h]
  __int64 v42; // [rsp+18h] [rbp-68h]
  __int64 v43; // [rsp+18h] [rbp-68h]
  char *v44; // [rsp+20h] [rbp-60h]
  __int64 v45; // [rsp+20h] [rbp-60h]
  unsigned __int64 v46; // [rsp+20h] [rbp-60h]
  void *v47; // [rsp+28h] [rbp-58h]
  char *v48; // [rsp+28h] [rbp-58h]
  __int64 v49; // [rsp+28h] [rbp-58h]
  void *src; // [rsp+30h] [rbp-50h] BYREF
  char *v51; // [rsp+38h] [rbp-48h]
  char *v52; // [rsp+40h] [rbp-40h]

  v9 = a4;
  v10 = &a4[2 * a5];
  src = 0;
  v51 = 0;
  v52 = 0;
  sub_ED2FF0((__int64)&src, a5);
  for ( ; v10 != v9; v52 = (char *)v11 )
  {
    while ( 1 )
    {
      v14 = sub_ED5180(a1, *v9, a2, a6, v11, (__int64)v12);
      v13 = v51;
      v15 = v52;
      v16 = v9[1];
      v17 = v14;
      if ( v51 == v52 )
        break;
      if ( v51 )
      {
        *(_QWORD *)v51 = v14;
        *((_QWORD *)v13 + 1) = v16;
        v13 = v51;
      }
      v9 += 2;
      v51 = v13 + 16;
      if ( v10 == v9 )
        goto LABEL_18;
    }
    v18 = src;
    v19 = v51 - (_BYTE *)src;
    v20 = (v51 - (_BYTE *)src) >> 4;
    if ( v20 == 0x7FFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"vector::_M_realloc_insert");
    v21 = 1;
    if ( v20 )
      v21 = v19 >> 4;
    v22 = __CFADD__(v21, v20);
    v23 = v21 + v20;
    if ( v22 )
    {
      v37 = 0x7FFFFFFFFFFFFFF0LL;
    }
    else
    {
      if ( !v23 )
      {
        v11 = 0;
        v12 = 0;
        goto LABEL_13;
      }
      if ( v23 > 0x7FFFFFFFFFFFFFFLL )
        v23 = 0x7FFFFFFFFFFFFFFLL;
      v37 = 16 * v23;
    }
    v39 = v51 - (_BYTE *)src;
    v41 = src;
    v46 = v9[1];
    v49 = v37;
    v38 = sub_22077B0(v37);
    v15 = v52;
    v16 = v46;
    v12 = (char *)v38;
    v18 = v41;
    v19 = v39;
    v11 = v38 + v49;
LABEL_13:
    v24 = (__int64 *)&v12[v19];
    if ( &v12[v19] )
    {
      *v24 = v17;
      v24[1] = v16;
    }
    v25 = (__int64)&v12[v19 + 16];
    if ( v19 > 0 )
    {
      v40 = v11;
      v42 = (__int64)&v12[v19 + 16];
      v44 = v15;
      v47 = v18;
      v36 = (char *)memmove(v12, v18, v19);
      v18 = v47;
      v15 = v44;
      v25 = v42;
      v11 = v40;
      v12 = v36;
LABEL_31:
      v43 = v11;
      v45 = v25;
      v48 = v12;
      j_j___libc_free_0(v18, v15 - (_BYTE *)v18);
      v11 = v43;
      v25 = v45;
      v12 = v48;
      goto LABEL_17;
    }
    if ( v18 )
      goto LABEL_31;
LABEL_17:
    v9 += 2;
    src = v12;
    v51 = (char *)v25;
  }
LABEL_18:
  v26 = a1;
  result = sub_ED1160(a1, a2);
  v29 = *(_QWORD **)(result + 8);
  v30 = result;
  if ( v29 == *(_QWORD **)(result + 16) )
  {
    result = sub_ED3260(result, *(char **)(result + 8), (__int64)&src);
    v34 = src;
  }
  else
  {
    if ( v29 )
    {
      v29[2] = 0;
      v31 = v51;
      v29[1] = 0;
      v32 = v31 - (_BYTE *)src;
      *v29 = 0;
      if ( v32 )
      {
        if ( v32 > 0x7FFFFFFFFFFFFFF0LL )
          sub_4261EA(v26, a2, v28);
        v33 = (char *)sub_22077B0(v32);
      }
      else
      {
        v32 = 0;
        v33 = 0;
      }
      *v29 = v33;
      v34 = src;
      v29[1] = v33;
      result = (__int64)v51;
      v29[2] = &v33[v32];
      v35 = result - (_QWORD)v34;
      if ( (void *)result != v34 )
      {
        result = (__int64)memmove(v33, v34, result - (_QWORD)v34);
        v33 = (char *)result;
      }
      v29[1] = &v33[v35];
      v29 = *(_QWORD **)(v30 + 8);
    }
    else
    {
      v34 = src;
    }
    *(_QWORD *)(v30 + 8) = v29 + 3;
  }
  if ( v34 )
    return j_j___libc_free_0(v34, v52 - (_BYTE *)v34);
  return result;
}
