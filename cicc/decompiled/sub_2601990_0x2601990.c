// Function: sub_2601990
// Address: 0x2601990
//
unsigned __int64 __fastcall sub_2601990(
        __int64 *a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char *a6,
        __int64 a7)
{
  char *v7; // r13
  __int64 v8; // rax
  __int64 *v9; // r14
  size_t v10; // r15
  unsigned __int64 result; // rax
  char *v12; // rsi
  __int64 *v13; // r13
  __int64 *v14; // r15
  __int64 v15; // rbx
  __int64 v16; // r12
  unsigned int v17; // r14d
  unsigned __int64 v18; // r8
  unsigned int v19; // r14d
  int v20; // eax
  unsigned int v21; // r14d
  __int64 v22; // rbx
  __int64 v23; // r12
  char *v24; // rcx
  char *v25; // rcx
  __int64 *v26; // r13
  __int64 *v27; // r12
  unsigned __int64 v28; // r8
  unsigned int v29; // r15d
  __int64 v30; // rbx
  __int64 v31; // r14
  unsigned int v32; // r15d
  int v33; // eax
  int v34; // eax
  unsigned int v35; // r15d
  __int64 *v36; // r12
  char *v37; // rax
  __int64 v38; // r10
  __int64 v39; // rcx
  char *v40; // r11
  __int64 v41; // r8
  __int64 v42; // r14
  char *v43; // r15
  int v44; // eax
  __int64 *v45; // rax
  char *v46; // rcx
  int v47; // [rsp+10h] [rbp-60h]
  __int64 v48; // [rsp+18h] [rbp-58h]
  __int64 v49; // [rsp+18h] [rbp-58h]
  char *v50; // [rsp+20h] [rbp-50h]
  __int64 v51; // [rsp+20h] [rbp-50h]
  __int64 v52; // [rsp+20h] [rbp-50h]
  __int64 *v53; // [rsp+20h] [rbp-50h]
  void *src; // [rsp+28h] [rbp-48h]
  char *v55; // [rsp+30h] [rbp-40h]
  char *v56; // [rsp+30h] [rbp-40h]
  __int64 v57; // [rsp+30h] [rbp-40h]
  __int64 v58; // [rsp+38h] [rbp-38h]
  char *v59; // [rsp+38h] [rbp-38h]
  unsigned __int64 v60; // [rsp+38h] [rbp-38h]

  while ( 1 )
  {
    v7 = a6;
    v58 = a3;
    v8 = a7;
    if ( a5 <= a7 )
      v8 = a5;
    if ( a4 <= v8 )
    {
      v9 = a2;
      v10 = (char *)a2 - (char *)a1;
      if ( a2 != a1 )
        memmove(a6, a1, v10);
      result = (unsigned __int64)&v7[v10];
      v55 = &v7[v10];
      if ( &v7[v10] == v7 )
        return result;
      v12 = v7;
      v13 = a1;
      v14 = v9;
      while ( 1 )
      {
        if ( (__int64 *)v58 == v14 )
          return (unsigned __int64)memmove(v13, v12, v55 - v12);
        v15 = *v14;
        v16 = *(_QWORD *)v12;
        v17 = *(_DWORD *)(*v14 + 32);
        if ( v17 > 0x40 )
        {
          v44 = sub_C444A0(v15 + 24);
          v18 = -1;
          if ( v17 - v44 <= 0x40 )
            v18 = **(_QWORD **)(v15 + 24);
        }
        else
        {
          v18 = *(_QWORD *)(v15 + 24);
        }
        v19 = *(_DWORD *)(v16 + 32);
        if ( v19 <= 0x40 )
          break;
        src = (void *)v18;
        v20 = sub_C444A0(v16 + 24);
        v18 = (unsigned __int64)src;
        v21 = v19 - v20;
        result = -1;
        if ( v21 > 0x40 )
        {
LABEL_9:
          if ( result > v18 )
            goto LABEL_10;
LABEL_17:
          v12 += 8;
          *v13++ = v16;
          if ( v55 == v12 )
            return result;
        }
        else
        {
          result = **(_QWORD **)(v16 + 24);
          if ( result <= (unsigned __int64)src )
            goto LABEL_17;
LABEL_10:
          *v13 = v15;
          ++v14;
          ++v13;
          if ( v55 == v12 )
            return result;
        }
      }
      result = *(_QWORD *)(v16 + 24);
      goto LABEL_9;
    }
    v22 = a5;
    if ( a5 <= a7 )
      break;
    v48 = a4;
    if ( a4 > a5 )
    {
      v57 = a4 / 2;
      v53 = &a1[a4 / 2];
      v45 = sub_25F6FF0(a2, a3, v53);
      v40 = (char *)v53;
      v39 = v48;
      v38 = a7;
      v36 = v45;
      v41 = v45 - a2;
    }
    else
    {
      v51 = a5 / 2;
      v36 = &a2[a5 / 2];
      v37 = (char *)sub_25F7100(a1, (__int64)a2, v36);
      v38 = a7;
      v39 = v48;
      v40 = v37;
      v41 = v51;
      v57 = (v37 - (char *)a1) >> 3;
    }
    v42 = v39 - v57;
    v49 = v38;
    v52 = v41;
    v47 = (int)v40;
    v43 = sub_2601870(v40, (char *)a2, (char *)v36, v39 - v57, v41, v7, v38);
    sub_2601990((_DWORD)a1, v47, (_DWORD)v43, v57, v52, (_DWORD)v7, v49);
    a6 = v7;
    a4 = v42;
    a2 = v36;
    a7 = v49;
    a3 = v58;
    a5 = v22 - v52;
    a1 = (__int64 *)v43;
  }
  result = a3;
  v23 = a3 - (_QWORD)a2;
  if ( (__int64 *)a3 != a2 )
    result = (unsigned __int64)memmove(a6, a2, a3 - (_QWORD)a2);
  v24 = &v7[v23];
  if ( a2 != a1 )
  {
    if ( v7 == v24 )
      return result;
    v50 = v7;
    v25 = v24 - 8;
    v26 = a2 - 1;
    v27 = (__int64 *)(v58 - 8);
    while ( 2 )
    {
      v30 = *(_QWORD *)v25;
      v31 = *v26;
      v32 = *(_DWORD *)(*(_QWORD *)v25 + 32LL);
      if ( v32 <= 0x40 )
      {
        v28 = *(_QWORD *)(v30 + 24);
      }
      else
      {
        v59 = v25;
        v33 = sub_C444A0(v30 + 24);
        v25 = v59;
        v28 = -1;
        if ( v32 - v33 <= 0x40 )
        {
          v29 = *(_DWORD *)(v31 + 32);
          v28 = **(_QWORD **)(v30 + 24);
          if ( v29 > 0x40 )
          {
LABEL_35:
            v56 = v25;
            v60 = v28;
            v34 = sub_C444A0(v31 + 24);
            v28 = v60;
            v25 = v56;
            v35 = v29 - v34;
            result = -1;
            if ( v35 <= 0x40 )
            {
              result = **(_QWORD **)(v31 + 24);
              if ( result <= v60 )
              {
LABEL_37:
                *v27 = v30;
                if ( v50 == v25 )
                  return result;
                v25 -= 8;
LABEL_31:
                --v27;
                continue;
              }
LABEL_29:
              *v27 = v31;
              if ( a1 == v26 )
              {
                v46 = v25 + 8;
                if ( v50 != v46 )
                  return (unsigned __int64)memmove((char *)v27 - (v46 - v50), v50, v46 - v50);
                return result;
              }
              --v26;
              goto LABEL_31;
            }
LABEL_28:
            if ( result <= v28 )
              goto LABEL_37;
            goto LABEL_29;
          }
LABEL_27:
          result = *(_QWORD *)(v31 + 24);
          goto LABEL_28;
        }
      }
      break;
    }
    v29 = *(_DWORD *)(v31 + 32);
    if ( v29 > 0x40 )
      goto LABEL_35;
    goto LABEL_27;
  }
  if ( v7 != v24 )
    return (unsigned __int64)memmove(a2, v7, v58 - (_QWORD)a2);
  return result;
}
