// Function: sub_14785F0
// Address: 0x14785f0
//
__int64 __fastcall sub_14785F0(__int64 a1, __int64 **a2, __int64 a3, unsigned int a4)
{
  __int64 v7; // rax
  __int64 *v8; // rdx
  __int64 result; // rax
  __int64 v10; // rcx
  __int64 v11; // r8
  _QWORD *v12; // rax
  _QWORD *v13; // rax
  unsigned __int64 v14; // r15
  __int64 v15; // rsi
  __int16 v16; // dx
  unsigned __int64 v17; // rdi
  _QWORD *v18; // rax
  _QWORD *v19; // rdx
  unsigned int v20; // ecx
  unsigned int v21; // eax
  _BYTE *v22; // rsi
  _BYTE *v23; // rdx
  __int64 v24; // r8
  __int64 *v25; // r15
  __int64 v26; // rax
  __int64 *v27; // rcx
  __int64 v28; // rax
  void *v29; // rax
  __int64 *v30; // r9
  size_t v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rbx
  __int64 v34; // rdx
  __int64 v35; // r14
  __int64 v36; // rax
  signed __int64 v37; // rax
  _QWORD *v38; // r15
  __int64 v39; // r8
  __int64 *v40; // r15
  __int64 v41; // rax
  __int64 *v42; // rcx
  __int64 v43; // rax
  signed __int64 v44; // rax
  __int64 *v45; // [rsp+0h] [rbp-100h]
  __int64 *v46; // [rsp+0h] [rbp-100h]
  __int64 *v47; // [rsp+8h] [rbp-F8h]
  __int64 *v48; // [rsp+8h] [rbp-F8h]
  __int64 *v49; // [rsp+8h] [rbp-F8h]
  __int64 v50; // [rsp+10h] [rbp-F0h]
  __int64 v51; // [rsp+10h] [rbp-F0h]
  __int16 v52; // [rsp+1Ch] [rbp-E4h]
  __int64 v53; // [rsp+28h] [rbp-D8h]
  __int64 v54; // [rsp+28h] [rbp-D8h]
  __int64 v55; // [rsp+28h] [rbp-D8h]
  void *v56; // [rsp+28h] [rbp-D8h]
  __int64 v57; // [rsp+28h] [rbp-D8h]
  __int64 v58; // [rsp+38h] [rbp-C8h] BYREF
  _BYTE *v59; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v60; // [rsp+48h] [rbp-B8h]
  _BYTE v61[176]; // [rsp+50h] [rbp-B0h] BYREF

  while ( 1 )
  {
    v7 = *((unsigned int *)a2 + 2);
    v8 = *a2;
    if ( v7 == 1 )
      return *v8;
    if ( !sub_14560B0(v8[v7 - 1]) )
      break;
    --*((_DWORD *)a2 + 2);
    a4 = 0;
  }
  v52 = sub_1478130(a1, 7, a2, a4);
  v50 = **a2;
  if ( *(_WORD *)(v50 + 24) != 7 )
    goto LABEL_16;
  v12 = *(_QWORD **)(v50 + 48);
  v53 = (__int64)v12;
  if ( (_QWORD *)a3 == v12 )
  {
LABEL_23:
    v18 = *(_QWORD **)a3;
    v19 = *(_QWORD **)v53;
    if ( *(_QWORD *)a3 )
    {
      v20 = 1;
      do
      {
        v18 = (_QWORD *)*v18;
        ++v20;
      }
      while ( v18 );
      if ( !v19 )
      {
        v21 = 1;
        goto LABEL_29;
      }
    }
    else
    {
      v20 = 1;
      if ( !v19 )
        goto LABEL_16;
    }
    v21 = 1;
    do
    {
      v19 = (_QWORD *)*v19;
      ++v21;
    }
    while ( v19 );
LABEL_29:
    if ( v20 >= v21 )
      goto LABEL_16;
    goto LABEL_30;
  }
  while ( v12 )
  {
    v12 = (_QWORD *)*v12;
    if ( (_QWORD *)a3 == v12 )
      goto LABEL_23;
  }
  if ( v53 == a3 )
  {
LABEL_16:
    v59 = v61;
    v60 = 0x2000000000LL;
    sub_16BD3E0(&v59, 7);
    v14 = 0;
    v54 = 8LL * *((unsigned int *)a2 + 2);
    if ( *((_DWORD *)a2 + 2) )
    {
      do
      {
        v15 = (*a2)[v14 / 8];
        v14 += 8LL;
        sub_16BD4C0(&v59, v15);
      }
      while ( v54 != v14 );
    }
    sub_16BD4C0(&v59, a3);
    v58 = 0;
    result = sub_16BDDE0(a1 + 816, &v59, &v58);
    if ( !result )
    {
      v29 = (void *)sub_145CBF0((__int64 *)(a1 + 864), 8LL * *((unsigned int *)a2 + 2), 8);
      v30 = (__int64 *)(a1 + 864);
      v56 = v29;
      v31 = 8LL * *((unsigned int *)a2 + 2);
      if ( v31 )
      {
        memmove(v29, *a2, v31);
        v30 = (__int64 *)(a1 + 864);
      }
      v48 = v30;
      v32 = sub_16BD760(&v59, v30);
      v33 = *((unsigned int *)a2 + 2);
      v51 = v32;
      v35 = v34;
      v36 = sub_145CDC0(0x38u, v48);
      if ( v36 )
      {
        *(_QWORD *)v36 = 0;
        *(_QWORD *)(v36 + 16) = v35;
        *(_QWORD *)(v36 + 8) = v51;
        *(_DWORD *)(v36 + 24) = 7;
        *(_QWORD *)(v36 + 32) = v56;
        *(_QWORD *)(v36 + 40) = v33;
        *(_QWORD *)(v36 + 48) = a3;
      }
      v57 = v36;
      sub_16BDA20(a1 + 816, v36, v58);
      sub_146DBF0(a1, v57);
      result = v57;
    }
    v16 = v52 | 1;
    if ( (v52 & 6) == 0 )
      v16 = v52;
    *(_WORD *)(result + 26) |= v16;
    v17 = (unsigned __int64)v59;
    if ( v59 != v61 )
      goto LABEL_22;
    return result;
  }
  if ( a3 )
  {
    v13 = (_QWORD *)a3;
    do
    {
      v13 = (_QWORD *)*v13;
      if ( (_QWORD *)v53 == v13 )
        goto LABEL_16;
    }
    while ( v13 );
  }
  if ( !(unsigned __int8)sub_15CC8F0(*(_QWORD *)(a1 + 56), **(_QWORD **)(a3 + 32), **(_QWORD **)(v53 + 32), v10, v11) )
    goto LABEL_16;
LABEL_30:
  v22 = *(_BYTE **)(v50 + 32);
  v23 = &v22[8 * *(_QWORD *)(v50 + 40)];
  v59 = v61;
  v60 = 0x400000000LL;
  sub_145C5B0((__int64)&v59, v22, v23);
  **a2 = **(_QWORD **)(v50 + 32);
  v25 = *a2;
  v26 = *((unsigned int *)a2 + 2);
  v27 = &(*a2)[v26];
  v28 = (v26 * 8) >> 5;
  v45 = v27;
  if ( !v28 )
  {
LABEL_45:
    v37 = (char *)v45 - (char *)v25;
    if ( (char *)v45 - (char *)v25 != 16 )
    {
      if ( v37 != 24 )
      {
        if ( v37 != 8 )
          goto LABEL_49;
        goto LABEL_48;
      }
      if ( !sub_146CEE0(a1, *v25, a3) )
        goto LABEL_37;
      ++v25;
    }
    if ( !sub_146CEE0(a1, *v25, a3) )
      goto LABEL_37;
    ++v25;
LABEL_48:
    if ( sub_146CEE0(a1, *v25, a3) )
      goto LABEL_49;
    goto LABEL_37;
  }
  v47 = &v25[4 * v28];
  while ( sub_146CEE0(a1, *v25, a3) )
  {
    if ( !sub_146CEE0(a1, v25[1], a3) )
    {
      ++v25;
      break;
    }
    if ( !sub_146CEE0(a1, v25[2], a3) )
    {
      v25 += 2;
      break;
    }
    if ( !sub_146CEE0(a1, v25[3], a3) )
    {
      v25 += 3;
      break;
    }
    v25 += 4;
    if ( v47 == v25 )
      goto LABEL_45;
  }
LABEL_37:
  if ( v45 != v25 )
    goto LABEL_38;
LABEL_49:
  v38 = v59;
  *v38 = sub_14785F0(a1, a2, a3, (unsigned __int8)v52 & (*(_WORD *)(v50 + 26) & 6 | 1u), v24);
  v40 = (__int64 *)v59;
  v41 = 8LL * (unsigned int)v60;
  v42 = (__int64 *)&v59[v41];
  v43 = v41 >> 5;
  v46 = v42;
  if ( !v43 )
  {
LABEL_71:
    v44 = (char *)v46 - (char *)v40;
    if ( (char *)v46 - (char *)v40 != 16 )
    {
      if ( v44 != 24 )
      {
        if ( v44 != 8 )
          goto LABEL_57;
        goto LABEL_74;
      }
      if ( !sub_146CEE0(a1, *v40, v53) )
        goto LABEL_56;
      ++v40;
    }
    if ( !sub_146CEE0(a1, *v40, v53) )
      goto LABEL_56;
    ++v40;
LABEL_74:
    if ( sub_146CEE0(a1, *v40, v53) )
      goto LABEL_57;
    goto LABEL_56;
  }
  v49 = (__int64 *)&v59[32 * v43];
  while ( sub_146CEE0(a1, *v40, v53) )
  {
    if ( !sub_146CEE0(a1, v40[1], v53) )
    {
      ++v40;
      break;
    }
    if ( !sub_146CEE0(a1, v40[2], v53) )
    {
      v40 += 2;
      break;
    }
    if ( !sub_146CEE0(a1, v40[3], v53) )
    {
      v40 += 3;
      break;
    }
    v40 += 4;
    if ( v49 == v40 )
      goto LABEL_71;
  }
LABEL_56:
  if ( v46 != v40 )
  {
LABEL_38:
    **a2 = v50;
    if ( v59 != v61 )
      _libc_free((unsigned __int64)v59);
    goto LABEL_16;
  }
LABEL_57:
  result = sub_14785F0(a1, &v59, v53, ((unsigned __int8)v52 | 1) & *(_WORD *)(v50 + 26) & 7u, v39);
  v17 = (unsigned __int64)v59;
  if ( v59 != v61 )
  {
LABEL_22:
    v55 = result;
    _libc_free(v17);
    return v55;
  }
  return result;
}
