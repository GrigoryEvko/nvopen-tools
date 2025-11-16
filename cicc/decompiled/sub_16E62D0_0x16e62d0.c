// Function: sub_16E62D0
// Address: 0x16e62d0
//
void __fastcall sub_16E62D0(__int64 a1, __int64 *a2, unsigned int a3, unsigned __int64 a4, char a5)
{
  unsigned int v7; // ebx
  char *v8; // r9
  __int64 *v10; // r15
  __int64 v11; // rdx
  unsigned int v12; // edi
  unsigned __int64 v13; // rax
  __int64 v14; // rcx
  bool v15; // cf
  unsigned __int64 v16; // rax
  char *v17; // rax
  const void *v18; // rsi
  size_t v19; // rdx
  __int64 v20; // r8
  __int64 *v21; // rdi
  __int64 *v22; // r9
  __int64 v23; // rsi
  __int64 v24; // rcx
  __int64 v25; // rax
  int v26; // edx
  bool v27; // sf
  __int64 v28; // r14
  __int64 v29; // rax
  __int64 *v30; // r9
  __int64 v31; // r10
  int v32; // r14d
  size_t v33; // rdx
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 *v37; // r8
  __int64 v38; // rdx
  __int64 v39; // rsi
  __int64 v40; // r11
  __int64 v41; // rax
  __int64 v42; // rcx
  __int64 v43; // r11
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rdx
  char *v47; // r9
  __int64 v48; // r8
  size_t v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r14
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // r14
  __int64 v55; // rax
  __int64 v56; // rcx
  __int64 v57; // rax
  __int64 v58; // rcx
  unsigned __int64 v59; // rdx
  unsigned __int64 v60; // rax
  unsigned __int64 v61; // rdx
  unsigned __int64 v62; // rdx
  unsigned __int64 v63; // rax
  unsigned __int64 v64; // rdx
  __int64 v65; // [rsp+10h] [rbp-70h]
  __int64 v66; // [rsp+10h] [rbp-70h]
  char *v67; // [rsp+18h] [rbp-68h]
  char v68; // [rsp+18h] [rbp-68h]
  char v69; // [rsp+18h] [rbp-68h]
  unsigned __int64 v70; // [rsp+20h] [rbp-60h]
  __int64 v71; // [rsp+20h] [rbp-60h]
  __int64 v72; // [rsp+20h] [rbp-60h]
  __int64 *v74; // [rsp+28h] [rbp-58h]
  char *v75; // [rsp+28h] [rbp-58h]
  __int64 *v76; // [rsp+28h] [rbp-58h]
  char *v77; // [rsp+28h] [rbp-58h]
  __int64 *v79; // [rsp+40h] [rbp-40h]

  if ( !a4 )
    return;
  v7 = a3;
  v8 = *(char **)(a1 + 16);
  v10 = a2;
  v11 = *(_QWORD *)(a1 + 32) - *(_QWORD *)a1;
  v12 = *(_DWORD *)(a1 + 24);
  v13 = v12 + 8LL * (_QWORD)&v8[-*(_QWORD *)a1];
  if ( 8 * v11 - v13 >= a4 )
  {
    v36 = a4 + v12 + 63;
    if ( (__int64)(a4 + v12) >= 0 )
      v36 = a4 + v12;
    v37 = (__int64 *)&v8[8 * (v36 >> 6)];
    v38 = (__int64)(a4 + v12) % 64;
    if ( v38 < 0 )
    {
      LODWORD(v38) = v38 + 64;
      --v37;
    }
    v39 = v12 + 8 * (v8 - (char *)a2) - a3;
    if ( v12 + 8 * (v8 - (char *)a2) - a3 <= 0 )
    {
LABEL_56:
      v45 = a4 + v7;
      v46 = v45 + 63;
      if ( v45 >= 0 )
        v46 = a4 + v7;
      v47 = (char *)&a2[v46 >> 6];
      v48 = v45 % 64;
      if ( v45 % 64 < 0 )
      {
        v48 += 64;
        v47 -= 8;
      }
      if ( v47 == (char *)a2 )
      {
        if ( (_DWORD)v48 != a3 )
        {
          v62 = (-1LL << a3) & (0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v48));
          v63 = *(_QWORD *)v47 | v62;
          v64 = *(_QWORD *)v47 & ~v62;
          if ( !a5 )
            v63 = v64;
          *(_QWORD *)v47 = v63;
        }
        goto LABEL_65;
      }
      if ( a3 )
      {
        v10 = a2 + 1;
        v55 = -1LL << a3;
        v49 = v47 - (char *)(a2 + 1);
        v56 = *a2;
        if ( a5 )
        {
          *a2 = v56 | v55;
LABEL_63:
          v68 = v48;
          v71 = v48;
          v75 = v47;
          memset(v10, -1, v49);
          if ( v71 )
            *(_QWORD *)v75 |= 0xFFFFFFFFFFFFFFFFLL >> (64 - v68);
LABEL_65:
          v50 = *(unsigned int *)(a1 + 24);
          v27 = (__int64)(v50 + a4) < 0;
          v51 = v50 + a4;
          v52 = v51 + 63;
          if ( !v27 )
            v52 = v51;
          v53 = *(_QWORD *)(a1 + 16) + 8 * (v52 >> 6);
          v54 = v51 % 64;
          if ( v54 < 0 )
          {
            LODWORD(v54) = v54 + 64;
            v53 -= 8;
          }
          *(_QWORD *)(a1 + 16) = v53;
          *(_DWORD *)(a1 + 24) = v54;
          return;
        }
        *a2 = v56 & ~v55;
      }
      else
      {
        v49 = v47 - (char *)a2;
        if ( a5 )
          goto LABEL_63;
      }
      v69 = v48;
      v72 = v48;
      v77 = v47;
      memset(v10, 0, v49);
      if ( v72 )
        *(_QWORD *)v77 &= ~(0xFFFFFFFFFFFFFFFFLL >> (64 - v69));
      goto LABEL_65;
    }
    while ( 1 )
    {
      if ( v12 )
      {
        v40 = 1LL << --v12;
        if ( !(_DWORD)v38 )
          goto LABEL_55;
      }
      else
      {
        v8 -= 8;
        v40 = 0x8000000000000000LL;
        v12 = 63;
        if ( !(_DWORD)v38 )
        {
LABEL_55:
          --v37;
          v41 = 0x8000000000000000LL;
          LODWORD(v38) = 63;
          goto LABEL_50;
        }
      }
      LODWORD(v38) = v38 - 1;
      v41 = 1LL << v38;
LABEL_50:
      v42 = v40 & *(_QWORD *)v8;
      v43 = v41 | *v37;
      v44 = *v37 & ~v41;
      if ( v42 )
        v44 = v43;
      *v37 = v44;
      if ( !--v39 )
        goto LABEL_56;
    }
  }
  if ( a4 > 0x7FFFFFFFFFFFFFC0LL - v13 )
    sub_4262D8((__int64)"vector<bool>::_M_fill_insert");
  v14 = v12 + 8LL * (_QWORD)&v8[-*(_QWORD *)a1];
  if ( a4 >= v13 )
    v14 = a4;
  v15 = __CFADD__(v14, v13);
  v16 = v14 + v13;
  if ( v15 )
  {
    v70 = 0xFFFFFFFFFFFFFF8LL;
    v17 = (char *)sub_22077B0(0xFFFFFFFFFFFFFF8LL);
  }
  else
  {
    if ( v16 > 0x7FFFFFFFFFFFFFC0LL )
      v16 = 0x7FFFFFFFFFFFFFC0LL;
    v70 = 8 * ((v16 + 63) >> 6);
    v17 = (char *)sub_22077B0(v70);
  }
  v18 = *(const void **)a1;
  v67 = v17;
  v19 = (size_t)a2 - *(_QWORD *)a1;
  if ( *(__int64 **)a1 != a2 )
  {
    memmove(v17, v18, v19);
    v19 = (char *)a2 - (_BYTE *)v18;
  }
  v20 = a3;
  v21 = (__int64 *)&v67[v19];
  if ( a3 )
  {
    v22 = a2;
    v23 = a3;
    LODWORD(v24) = 0;
    do
    {
      while ( 1 )
      {
        v25 = *v21 & ~(1LL << v24);
        if ( ((1LL << v24) & *v22) != 0 )
          v25 = (1LL << v24) | *v21;
        *v21 = v25;
        if ( (_DWORD)v24 == 63 )
          break;
        LODWORD(v24) = v24 + 1;
        if ( !--v23 )
          goto LABEL_19;
      }
      ++v22;
      ++v21;
      LODWORD(v24) = 0;
      --v23;
    }
    while ( v23 );
LABEL_19:
    v24 = (unsigned int)v24;
    v26 = v24;
  }
  else
  {
    v24 = 0;
    v26 = 0;
  }
  v27 = (__int64)(v24 + a4) < 0;
  v28 = v24 + a4;
  v29 = v28 + 63;
  if ( !v27 )
    v29 = v28;
  v30 = &v21[v29 >> 6];
  v31 = v28 % 64;
  if ( v28 % 64 < 0 )
  {
    v31 += 64;
    --v30;
  }
  v32 = v31;
  if ( v21 == v30 )
  {
    if ( v26 != (_DWORD)v31 )
    {
      v59 = (-1LL << v26) & (0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v31));
      v60 = *v30 | v59;
      v61 = *v30 & ~v59;
      if ( !a5 )
        v60 = v61;
      *v30 = v60;
    }
    goto LABEL_29;
  }
  if ( !v26 )
  {
    v33 = (char *)v30 - (char *)v21;
    if ( a5 )
      goto LABEL_27;
LABEL_75:
    v66 = v31;
    v76 = v30;
    memset(v21, 0, v33);
    v30 = v76;
    v20 = v7;
    if ( v66 )
      *v76 &= ~(0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v32));
    goto LABEL_29;
  }
  v57 = -1LL << v26;
  v33 = (char *)v30 - (char *)(v21 + 1);
  v58 = *v21;
  if ( !a5 )
  {
    *v21++ = v58 & ~v57;
    goto LABEL_75;
  }
  *v21++ = v58 | v57;
LABEL_27:
  v65 = v31;
  v74 = v30;
  memset(v21, -1, v33);
  v30 = v74;
  v20 = v7;
  if ( v65 )
    *v74 |= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v32);
LABEL_29:
  v34 = *(unsigned int *)(a1 + 24) + 8LL * (*(_QWORD *)(a1 + 16) - (_QWORD)a2) - v20;
  if ( v34 > 0 )
  {
    while ( 1 )
    {
      v35 = *v30 & ~(1LL << v32);
      if ( ((1LL << v7) & *v10) != 0 )
        v35 = (1LL << v32) | *v30;
      *v30 = v35;
      if ( v7 == 63 )
      {
        ++v10;
        v7 = 0;
        if ( v32 == 63 )
          goto LABEL_37;
LABEL_32:
        ++v32;
        if ( !--v34 )
          break;
      }
      else
      {
        ++v7;
        if ( v32 != 63 )
          goto LABEL_32;
LABEL_37:
        ++v30;
        v32 = 0;
        if ( !--v34 )
          break;
      }
    }
  }
  v79 = v30;
  if ( *(_QWORD *)a1 )
  {
    j_j___libc_free_0(*(_QWORD *)a1, *(_QWORD *)(a1 + 32) - *(_QWORD *)a1);
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_DWORD *)(a1 + 24) = 0;
    *(_DWORD *)(a1 + 8) = 0;
  }
  *(_DWORD *)(a1 + 8) = 0;
  *(_QWORD *)a1 = v67;
  *(_QWORD *)(a1 + 32) = &v67[v70];
  *(_QWORD *)(a1 + 16) = v79;
  *(_DWORD *)(a1 + 24) = v32;
}
