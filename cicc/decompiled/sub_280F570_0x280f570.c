// Function: sub_280F570
// Address: 0x280f570
//
__int64 __fastcall sub_280F570(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __int64 *a5,
        __int64 *a6,
        __int64 *a7,
        char a8)
{
  __int64 v11; // r12
  __int64 v12; // rbx
  __int64 v14; // rax
  unsigned __int64 v15; // rdi
  int v16; // eax
  __int64 v17; // rdi
  __int64 v18; // rsi
  _QWORD *v19; // rax
  _QWORD *v20; // rdx
  __int64 v21; // rax
  int v22; // eax
  __int64 v23; // rcx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rcx
  unsigned __int64 v29; // rsi
  int v30; // eax
  __int64 v31; // rsi
  __int64 *v32; // rax
  __int64 *v33; // rdi
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rdi
  __int64 v37; // rax
  __int64 v38; // rdi
  char v39; // al
  __int64 v40; // r13
  __int64 v41; // rbx
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 *v46; // r9
  __int64 *v47; // rsi
  unsigned int v48; // eax
  __int64 *v49; // rax
  __int64 *v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // rbx
  __int64 v53; // r8
  __int64 v54; // r9
  __int64 v55; // rsi
  __int64 *v56; // rax
  __int64 v57; // rax
  char v58; // al
  __int64 *v59; // rdx
  __int64 v60; // rsi
  _QWORD *v61; // rax
  _QWORD *v62; // rax
  _QWORD *v63; // rax
  __int64 *v64; // rax
  __int64 v65; // [rsp+0h] [rbp-80h]
  __int64 v66; // [rsp+8h] [rbp-78h]
  __int64 v67; // [rsp+8h] [rbp-78h]
  __int64 v68; // [rsp+8h] [rbp-78h]
  __int64 *v69; // [rsp+8h] [rbp-78h]
  __int64 *v71; // [rsp+20h] [rbp-60h]
  __int64 v73; // [rsp+28h] [rbp-58h]
  __int64 *v74; // [rsp+28h] [rbp-58h]
  __int64 v75; // [rsp+28h] [rbp-58h]
  __int64 v76; // [rsp+28h] [rbp-58h]
  __int64 *v77; // [rsp+28h] [rbp-58h]
  unsigned __int64 v78; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v79; // [rsp+38h] [rbp-48h]
  unsigned __int64 v80; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v81; // [rsp+48h] [rbp-38h]

  v11 = a1;
  v12 = sub_D47930(a1);
  if ( v12 != sub_D46F00(a1) )
    goto LABEL_2;
  v14 = sub_D4B410(a1, (__int64)a7);
  *(_QWORD *)a3 = v14;
  if ( !v14 )
    goto LABEL_2;
  v15 = *(_QWORD *)(v12 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v12 + 48 == v15 )
  {
    v17 = 0;
  }
  else
  {
    if ( !v15 )
      BUG();
    v16 = *(unsigned __int8 *)(v15 - 24);
    v17 = v15 - 24;
    if ( (unsigned int)(v16 - 30) >= 0xB )
      v17 = 0;
  }
  v18 = sub_B46EC0(v17, 0);
  if ( *(_BYTE *)(v11 + 84) )
  {
    v19 = *(_QWORD **)(v11 + 64);
    v20 = &v19[*(unsigned int *)(v11 + 76)];
    if ( v19 == v20 )
    {
LABEL_16:
      v24 = sub_D48970(v11);
      if ( !v24 )
        goto LABEL_2;
      v67 = v24;
      v22 = sub_B52EF0(*(_WORD *)(v24 + 2) & 0x3F);
      v23 = v67;
      goto LABEL_18;
    }
    while ( v18 != *v19 )
    {
      if ( v20 == ++v19 )
        goto LABEL_16;
    }
    v21 = sub_D48970(v11);
    if ( !v21 )
      goto LABEL_2;
    v66 = v21;
    v22 = sub_B52EF0(*(_WORD *)(v21 + 2) & 0x3F);
    v23 = v66;
  }
  else
  {
    v69 = sub_C8CA60(v11 + 56, v18);
    v57 = sub_D48970(v11);
    if ( !v57 )
      goto LABEL_2;
    v65 = v57;
    v22 = sub_B52EF0(*(_WORD *)(v57 + 2) & 0x3F);
    v23 = v65;
    if ( !v69 )
    {
LABEL_18:
      if ( v22 != 32 )
        goto LABEL_2;
      goto LABEL_19;
    }
  }
  if ( v22 != 33 && v22 != 36 )
    goto LABEL_2;
LABEL_19:
  v68 = v23;
  if ( (unsigned __int8)sub_BD3660(v23, 2) )
    goto LABEL_2;
  v28 = v68;
  v29 = *(_QWORD *)(v12 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v12 + 48 == v29 )
  {
    v31 = 0;
  }
  else
  {
    if ( !v29 )
      BUG();
    v30 = *(unsigned __int8 *)(v29 - 24);
    v31 = v29 - 24;
    if ( (unsigned int)(v30 - 30) >= 0xB )
      v31 = 0;
  }
  *a6 = v31;
  if ( !*(_BYTE *)(a2 + 28) )
    goto LABEL_65;
  v25 = *(unsigned int *)(a2 + 20);
  v32 = *(__int64 **)(a2 + 8);
  v33 = &v32[v25];
  v26 = v25;
  if ( v32 != v33 )
  {
    v25 = *(_QWORD *)(a2 + 8);
    while ( v31 != *(_QWORD *)v25 )
    {
      v25 += 8;
      if ( v33 == (__int64 *)v25 )
        goto LABEL_70;
    }
    goto LABEL_29;
  }
LABEL_70:
  if ( (unsigned int)v26 < *(_DWORD *)(a2 + 16) )
  {
    v26 = (unsigned int)(v26 + 1);
    *(_DWORD *)(a2 + 20) = v26;
    *v33 = v31;
    v32 = *(__int64 **)(a2 + 8);
    ++*(_QWORD *)a2;
    v34 = *(unsigned __int8 *)(a2 + 28);
  }
  else
  {
LABEL_65:
    sub_C8CC70(a2, v31, v25, v68, v26, v27);
    v34 = *(unsigned __int8 *)(a2 + 28);
    v32 = *(__int64 **)(a2 + 8);
    v28 = v68;
  }
  if ( !(_BYTE)v34 )
    goto LABEL_68;
  v26 = *(unsigned int *)(a2 + 20);
LABEL_29:
  v34 = (__int64)&v32[(unsigned int)v26];
  if ( v32 == (__int64 *)v34 )
  {
LABEL_72:
    if ( (unsigned int)v26 < *(_DWORD *)(a2 + 16) )
    {
      *(_DWORD *)(a2 + 20) = v26 + 1;
      *(_QWORD *)v34 = v28;
      ++*(_QWORD *)a2;
      goto LABEL_33;
    }
LABEL_68:
    v75 = v28;
    sub_C8CC70(a2, v28, v34, v28, v26, v27);
    v28 = v75;
    goto LABEL_33;
  }
  while ( *v32 != v28 )
  {
    if ( (__int64 *)v34 == ++v32 )
      goto LABEL_72;
  }
LABEL_33:
  v35 = 0x1FFFFFFFE0LL;
  v36 = *(_QWORD *)(*(_QWORD *)a3 - 8LL);
  if ( (*(_DWORD *)(*(_QWORD *)a3 + 4LL) & 0x7FFFFFF) != 0 )
  {
    v37 = 0;
    do
    {
      if ( v12 == *(_QWORD *)(v36 + 32LL * *(unsigned int *)(*(_QWORD *)a3 + 72LL) + 8 * v37) )
      {
        v35 = 32 * v37;
        goto LABEL_38;
      }
      ++v37;
    }
    while ( (*(_DWORD *)(*(_QWORD *)a3 + 4LL) & 0x7FFFFFF) != (_DWORD)v37 );
    v35 = 0x1FFFFFFFE0LL;
  }
LABEL_38:
  v38 = *(_QWORD *)(v36 + v35);
  *a5 = v38;
  if ( *(_QWORD *)(v28 - 64) == v38 )
  {
    v76 = v28;
    v58 = sub_BD3610(v38, 2);
    v28 = v76;
    if ( v58 )
      goto LABEL_40;
    v38 = *a5;
  }
  v73 = v28;
  v39 = sub_BD3610(v38, 1);
  v28 = v73;
  if ( !v39 )
    goto LABEL_2;
LABEL_40:
  v40 = *(_QWORD *)(v28 - 32);
  v41 = sub_DCF3A0(a7, (char *)v11, 0);
  if ( sub_D96A50(v41) )
    goto LABEL_2;
  v42 = sub_D95540(v41);
  v74 = sub_DE5A20(a7, v41, v42, v11);
  v46 = sub_DD8400((__int64)a7, v40);
  if ( v74 == v46 )
  {
    *a4 = v40;
    v60 = *a5;
    if ( !*(_BYTE *)(a2 + 28) )
      goto LABEL_88;
    v63 = *(_QWORD **)(a2 + 8);
    v44 = *(unsigned int *)(a2 + 20);
    v43 = (__int64)&v63[v44];
    if ( v63 != (_QWORD *)v43 )
    {
      while ( v60 != *v63 )
      {
        if ( (_QWORD *)v43 == ++v63 )
          goto LABEL_104;
      }
      goto LABEL_86;
    }
    goto LABEL_104;
  }
  if ( *(_BYTE *)v40 != 17 )
  {
    if ( (unsigned __int8)(*(_BYTE *)v40 - 68) <= 1u && a8 == 1 )
    {
      v59 = (*(_BYTE *)(v40 + 7) & 0x40) != 0
          ? *(__int64 **)(v40 - 8)
          : (__int64 *)(v40 - 32LL * (*(_DWORD *)(v40 + 4) & 0x7FFFFFF));
      if ( v74 == sub_DD8400((__int64)a7, *v59) )
      {
        *a4 = v40;
        v60 = *a5;
        if ( *(_BYTE *)(a2 + 28) )
        {
          v61 = *(_QWORD **)(a2 + 8);
          v43 = *(unsigned int *)(a2 + 20);
          v44 = (__int64)&v61[v43];
          if ( v61 != (_QWORD *)v44 )
          {
            while ( v60 != *v61 )
            {
              if ( (_QWORD *)v44 == ++v61 )
                goto LABEL_109;
            }
            goto LABEL_86;
          }
LABEL_109:
          if ( (unsigned int)v43 < *(_DWORD *)(a2 + 16) )
          {
            *(_DWORD *)(a2 + 20) = v43 + 1;
            *(_QWORD *)v44 = v60;
            ++*(_QWORD *)a2;
            goto LABEL_86;
          }
        }
        goto LABEL_88;
      }
    }
LABEL_2:
    LODWORD(v11) = 0;
    return (unsigned int)v11;
  }
  v47 = 0;
  if ( a8 )
  {
    v71 = v46;
    v77 = sub_DC2B70((__int64)a7, v41, *(_QWORD *)(v40 + 8), 0);
    v64 = sub_DE5A20(a7, (__int64)v77, *(_QWORD *)(v40 + 8), v11);
    v47 = v77;
    v46 = v71;
    if ( v71 != v77 && v71 != v64 )
      goto LABEL_2;
  }
  LOBYTE(v11) = v46 == v47 || v41 == (_QWORD)v46;
  if ( !(_BYTE)v11 )
  {
    *a4 = v40;
    v60 = *a5;
    if ( !*(_BYTE *)(a2 + 28) )
      goto LABEL_88;
    v62 = *(_QWORD **)(a2 + 8);
    v44 = *(unsigned int *)(a2 + 20);
    v43 = (__int64)&v62[v44];
    if ( v62 != (_QWORD *)v43 )
    {
      while ( v60 != *v62 )
      {
        if ( (_QWORD *)v43 == ++v62 )
          goto LABEL_104;
      }
      goto LABEL_86;
    }
LABEL_104:
    if ( (unsigned int)v44 < *(_DWORD *)(a2 + 16) )
    {
      *(_DWORD *)(a2 + 20) = v44 + 1;
      *(_QWORD *)v43 = v60;
      ++*(_QWORD *)a2;
      goto LABEL_86;
    }
LABEL_88:
    sub_C8CC70(a2, v60, v43, v44, v45, (__int64)v46);
LABEL_86:
    LODWORD(v11) = 1;
    return (unsigned int)v11;
  }
  v81 = *(_DWORD *)(v40 + 32);
  if ( v81 > 0x40 )
    sub_C43780((__int64)&v80, (const void **)(v40 + 24));
  else
    v80 = *(_QWORD *)(v40 + 24);
  sub_C46A40((__int64)&v80, 1);
  v48 = v81;
  v81 = 0;
  v79 = v48;
  v78 = v80;
  v49 = (__int64 *)sub_BD5C60(v40);
  v52 = sub_ACCFD0(v49, (__int64)&v78);
  if ( v79 > 0x40 && v78 )
    j_j___libc_free_0_0(v78);
  if ( v81 > 0x40 && v80 )
    j_j___libc_free_0_0(v80);
  *a4 = v52;
  v55 = *a5;
  if ( !*(_BYTE *)(a2 + 28) )
    goto LABEL_106;
  v56 = *(__int64 **)(a2 + 8);
  v51 = *(unsigned int *)(a2 + 20);
  v50 = &v56[v51];
  if ( v56 == v50 )
  {
LABEL_57:
    if ( (unsigned int)v51 < *(_DWORD *)(a2 + 16) )
    {
      *(_DWORD *)(a2 + 20) = v51 + 1;
      *v50 = v55;
      ++*(_QWORD *)a2;
      return (unsigned int)v11;
    }
LABEL_106:
    sub_C8CC70(a2, v55, (__int64)v50, v51, v53, v54);
    return (unsigned int)v11;
  }
  while ( v55 != *v56 )
  {
    if ( v50 == ++v56 )
      goto LABEL_57;
  }
  return (unsigned int)v11;
}
