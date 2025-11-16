// Function: sub_13E7B40
// Address: 0x13e7b40
//
__int64 __fastcall sub_13E7B40(__int64 a1, __int64 a2, __int64 a3)
{
  char *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rsi
  unsigned int v7; // ecx
  __int64 *v8; // rax
  __int64 v9; // r8
  _QWORD *v10; // rbx
  __int64 v11; // rax
  _QWORD *v12; // r12
  _BYTE *v13; // r8
  __int64 v14; // r13
  int v15; // eax
  _QWORD *v16; // rdx
  _QWORD *v17; // rax
  _QWORD *v18; // rcx
  __int64 v19; // rdx
  _QWORD *v20; // rax
  __int64 v21; // rbx
  __int64 v22; // rax
  __int64 v23; // rsi
  unsigned int v24; // ecx
  __int64 *v25; // r14
  int v26; // edx
  __int64 v27; // r9
  _BYTE *v28; // rbx
  _QWORD *v29; // r12
  _QWORD *v30; // rax
  __int64 v31; // rax
  __int64 v32; // r13
  _QWORD *v33; // rdx
  int v34; // eax
  __int64 v35; // rdi
  int v36; // ebx
  __int64 v37; // r12
  __int64 v38; // r13
  __int64 i; // r15
  __int64 result; // rax
  __int64 v41; // rdx
  __int64 v42; // rsi
  unsigned __int64 v43; // rdi
  size_t v44; // r14
  unsigned __int64 v45; // rax
  bool v46; // cf
  unsigned __int64 v47; // r13
  __int64 v48; // r15
  char *v49; // r13
  unsigned int v50; // r15d
  __int64 *v51; // r14
  __int64 v52; // rax
  int v53; // eax
  int v54; // r9d
  char *v55; // [rsp+10h] [rbp-A0h]
  __int64 v58; // [rsp+28h] [rbp-88h]
  char *src; // [rsp+30h] [rbp-80h]
  char *v60; // [rsp+38h] [rbp-78h]
  char *v61; // [rsp+38h] [rbp-78h]
  char *v62; // [rsp+40h] [rbp-70h]
  char v63; // [rsp+48h] [rbp-68h]
  char *v64; // [rsp+48h] [rbp-68h]
  _BYTE *v65; // [rsp+50h] [rbp-60h] BYREF
  __int64 v66; // [rsp+58h] [rbp-58h]
  _BYTE v67[80]; // [rsp+60h] [rbp-50h] BYREF

  v4 = (char *)sub_22077B0(8);
  src = v4;
  if ( v4 )
    *(_QWORD *)v4 = a2;
  v60 = v4 + 8;
  v5 = *(unsigned int *)(a1 + 88);
  if ( !(_DWORD)v5 )
    goto LABEL_96;
  v6 = *(_QWORD *)(a1 + 72);
  v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v6 + 80LL * v7);
  v9 = *v8;
  if ( *v8 != a2 )
  {
    v53 = 1;
    while ( v9 != -8 )
    {
      v54 = v53 + 1;
      v7 = (v5 - 1) & (v53 + v7);
      v8 = (__int64 *)(v6 + 80LL * v7);
      v9 = *v8;
      if ( *v8 == a2 )
        goto LABEL_5;
      v53 = v54;
    }
LABEL_96:
    result = (__int64)src;
    if ( src )
      return j_j___libc_free_0(src, 8);
    return result;
  }
LABEL_5:
  if ( v8 == (__int64 *)(v6 + 80 * v5) )
    goto LABEL_96;
  v10 = (_QWORD *)v8[3];
  if ( v10 == (_QWORD *)v8[2] )
    v11 = *((unsigned int *)v8 + 9);
  else
    v11 = *((unsigned int *)v8 + 8);
  v12 = &v10[v11];
  if ( v10 == v12 )
  {
LABEL_11:
    HIDWORD(v66) = 4;
    v65 = v67;
  }
  else
  {
    while ( *v10 >= 0xFFFFFFFFFFFFFFFELL )
    {
      if ( ++v10 == v12 )
        goto LABEL_11;
    }
    v65 = v67;
    v66 = 0x400000000LL;
    if ( v12 != v10 )
    {
      v16 = v10;
      v14 = 0;
      while ( 1 )
      {
        v17 = v16 + 1;
        if ( v16 + 1 == v12 )
          break;
        while ( 1 )
        {
          v16 = v17;
          if ( *v17 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v12 == ++v17 )
            goto LABEL_18;
        }
        ++v14;
        if ( v12 == v17 )
          goto LABEL_19;
      }
LABEL_18:
      ++v14;
LABEL_19:
      v18 = v67;
      if ( v14 > 4 )
      {
        sub_16CD150(&v65, v67, v14, 8);
        v18 = &v65[8 * (unsigned int)v66];
      }
      v19 = *v10;
      do
      {
        v20 = v10 + 1;
        *v18++ = v19;
        if ( v10 + 1 == v12 )
          break;
        while ( 1 )
        {
          v19 = *v20;
          v10 = v20;
          if ( *v20 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v12 == ++v20 )
            goto LABEL_25;
        }
      }
      while ( v20 != v12 );
LABEL_25:
      v15 = v66;
      v13 = v65;
      goto LABEL_26;
    }
  }
  v13 = v67;
  LODWORD(v14) = 0;
  v15 = 0;
LABEL_26:
  LODWORD(v66) = v14 + v15;
  v55 = v60;
  while ( 2 )
  {
    while ( 1 )
    {
      v21 = *((_QWORD *)v60 - 1);
      v62 = v60 - 8;
      v58 = v21;
      if ( v21 == a3 )
        break;
      v22 = *(unsigned int *)(a1 + 88);
      if ( !(_DWORD)v22 )
        break;
      v23 = *(_QWORD *)(a1 + 72);
      v24 = (v22 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v25 = (__int64 *)(v23 + 80LL * v24);
      v26 = 1;
      v27 = *v25;
      if ( v21 != *v25 )
      {
        while ( v27 != -8 )
        {
          v24 = (v22 - 1) & (v26 + v24);
          v25 = (__int64 *)(v23 + 80LL * v24);
          v27 = *v25;
          if ( v21 == *v25 )
            goto LABEL_30;
          ++v26;
        }
        break;
      }
LABEL_30:
      if ( v25 == (__int64 *)(v23 + 80 * v22) )
        break;
      v28 = &v13[8 * (unsigned int)v66];
      if ( v13 == v28 )
        break;
      v63 = 0;
      v29 = v13;
      do
      {
        while ( 1 )
        {
          v32 = *v29;
          v30 = (_QWORD *)v25[2];
          if ( (_QWORD *)v25[3] == v30 )
            break;
          v30 = (_QWORD *)sub_16CC9F0(v25 + 1, *v29);
          if ( v32 == *v30 )
          {
            v41 = v25[3];
            if ( v41 == v25[2] )
              v42 = *((unsigned int *)v25 + 9);
            else
              v42 = *((unsigned int *)v25 + 8);
            v33 = (_QWORD *)(v41 + 8 * v42);
            goto LABEL_41;
          }
          v31 = v25[3];
          if ( v31 == v25[2] )
          {
            v30 = (_QWORD *)(v31 + 8LL * *((unsigned int *)v25 + 9));
            v33 = v30;
            goto LABEL_41;
          }
LABEL_35:
          if ( v28 == (_BYTE *)++v29 )
            goto LABEL_44;
        }
        v33 = &v30[*((unsigned int *)v25 + 9)];
        if ( v30 == v33 )
        {
LABEL_59:
          v30 = v33;
        }
        else
        {
          while ( v32 != *v30 )
          {
            if ( v33 == ++v30 )
              goto LABEL_59;
          }
        }
LABEL_41:
        if ( v33 == v30 )
          goto LABEL_35;
        *v30 = -2;
        v34 = *((_DWORD *)v25 + 10) + 1;
        *((_DWORD *)v25 + 10) = v34;
        if ( *((_DWORD *)v25 + 9) == v34 )
        {
          v43 = v25[3];
          if ( v43 != v25[2] )
            _libc_free(v43);
          *v25 = -16;
          --*(_DWORD *)(a1 + 80);
          ++*(_DWORD *)(a1 + 84);
          goto LABEL_45;
        }
        ++v29;
        v63 = 1;
      }
      while ( v28 != (_BYTE *)v29 );
LABEL_44:
      v13 = v65;
      if ( !v63 )
        break;
LABEL_45:
      v35 = sub_157EBA0(v58);
      if ( v35 && (v36 = sub_15F4D60(v35), v37 = sub_157EBA0(v58), v36) )
      {
        v38 = v36;
        if ( v36 <= (unsigned __int64)((v55 - v62) >> 3) )
        {
          for ( i = 0; i != v36; ++i )
            *(_QWORD *)&v60[8 * i - 8] = sub_15F4DF0(v37, (unsigned int)i);
          v62 += 8 * v36;
          goto LABEL_51;
        }
        v44 = v62 - src;
        v45 = (v62 - src) >> 3;
        if ( v36 > 0xFFFFFFFFFFFFFFFLL - v45 )
          sub_4262D8((__int64)"vector::_M_range_insert");
        if ( v36 < v45 )
          v38 = (v62 - src) >> 3;
        v46 = __CFADD__(v45, v38);
        v47 = v45 + v38;
        if ( v46 )
        {
          v48 = 0x7FFFFFFFFFFFFFF8LL;
          goto LABEL_82;
        }
        if ( v47 )
        {
          if ( v47 > 0xFFFFFFFFFFFFFFFLL )
            v47 = 0xFFFFFFFFFFFFFFFLL;
          v48 = 8 * v47;
LABEL_82:
          v49 = (char *)sub_22077B0(v48);
          v61 = &v49[v48];
        }
        else
        {
          v61 = 0;
          v49 = 0;
        }
        if ( v62 != src )
          memmove(v49, src, v44);
        v50 = 0;
        v64 = &v49[v44];
        v51 = (__int64 *)&v49[v44];
        do
        {
          v52 = sub_15F4DF0(v37, v50);
          if ( v51 )
            *v51 = v52;
          ++v50;
          ++v51;
        }
        while ( v36 != v50 );
        v62 = &v64[8 * v36];
        if ( src )
          j_j___libc_free_0(src, v55 - src);
        v13 = v65;
        src = v49;
        v55 = v61;
        v60 = &v64[8 * v36];
      }
      else
      {
LABEL_51:
        v13 = v65;
        v60 = v62;
      }
      if ( src == v62 )
      {
        if ( v13 != v67 )
          goto LABEL_54;
        return j_j___libc_free_0(src, v55 - src);
      }
    }
    v60 -= 8;
    if ( src != v62 )
      continue;
    break;
  }
  if ( v13 != v67 )
LABEL_54:
    _libc_free((unsigned __int64)v13);
  return j_j___libc_free_0(src, v55 - src);
}
