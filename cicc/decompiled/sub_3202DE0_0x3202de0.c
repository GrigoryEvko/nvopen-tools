// Function: sub_3202DE0
// Address: 0x3202de0
//
void __fastcall sub_3202DE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 *v7; // r14
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned __int64 v10; // r15
  _BYTE *v11; // rdi
  int v12; // eax
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // rsi
  __int64 v16; // r15
  __int64 v17; // rdi
  __int64 v18; // r8
  __int64 v19; // r9
  unsigned __int64 v20; // rax
  __int64 *v21; // rsi
  __int64 v22; // rcx
  unsigned __int64 v23; // rcx
  __int64 *v24; // rdx
  unsigned __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rax
  int v28; // eax
  unsigned int v29; // ecx
  __int64 v30; // rdx
  _QWORD *v31; // rax
  _QWORD *j; // rdx
  __int64 v33; // r13
  __int64 v34; // rbx
  unsigned __int64 v35; // rdi
  unsigned __int64 v36; // rdi
  bool v37; // zf
  unsigned __int64 v38; // rdx
  __int64 *v39; // rax
  unsigned int v40; // eax
  _QWORD *v41; // rdi
  int v42; // ebx
  unsigned __int64 v43; // rdx
  unsigned __int64 v44; // rax
  _QWORD *v45; // rax
  __int64 v46; // rdx
  _QWORD *i; // rdx
  _QWORD *v48; // rax
  __int64 v49; // [rsp+10h] [rbp-A0h]
  __int64 v50; // [rsp+10h] [rbp-A0h]
  __int64 v51; // [rsp+18h] [rbp-98h]
  __int64 v54; // [rsp+38h] [rbp-78h] BYREF
  unsigned int v55; // [rsp+40h] [rbp-70h] BYREF
  unsigned __int64 v56; // [rsp+48h] [rbp-68h]
  unsigned int v57; // [rsp+50h] [rbp-60h]
  _BYTE v58[16]; // [rsp+58h] [rbp-58h] BYREF
  unsigned __int64 v59; // [rsp+68h] [rbp-48h]
  char v60; // [rsp+70h] [rbp-40h]
  char v61; // [rsp+78h] [rbp-38h]

  v5 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 232LL) + 16LL);
  v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 200LL))(v5);
  v7 = *(__int64 **)a3;
  v51 = v6;
  v8 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
  if ( (__int64 *)v8 == v7 )
    return;
  while ( 1 )
  {
    v9 = *v7;
    if ( (*v7 & 4) != 0 )
      goto LABEL_9;
    v10 = v9 & 0xFFFFFFFFFFFFFFF8LL;
    sub_32118A0(&v55, v9 & 0xFFFFFFFFFFFFFFF8LL);
    if ( v61 )
      break;
    v26 = *(_QWORD *)(v10 + 32);
    if ( *(_WORD *)(v10 + 68) != 14 )
      v26 += 80;
    if ( *(_BYTE *)v26 != 1 )
      goto LABEL_9;
    v27 = *(_QWORD *)(v26 + 24);
    if ( *(_BYTE *)(a2 + 80) )
    {
      if ( *(_DWORD *)(a2 + 72) > 0x40u && (v36 = *(_QWORD *)(a2 + 64)) != 0 )
      {
        v50 = *(_QWORD *)(v26 + 24);
        j_j___libc_free_0_0(v36);
        v37 = v61 == 0;
        *(_DWORD *)(a2 + 72) = 64;
        *(_BYTE *)(a2 + 76) = 0;
        *(_QWORD *)(a2 + 64) = v50;
        if ( !v37 )
        {
LABEL_29:
          v11 = (_BYTE *)v56;
          goto LABEL_7;
        }
      }
      else
      {
        *(_QWORD *)(a2 + 64) = v27;
        *(_DWORD *)(a2 + 72) = 64;
        *(_BYTE *)(a2 + 76) = 0;
      }
    }
    else
    {
      *(_DWORD *)(a2 + 72) = 64;
      *(_QWORD *)(a2 + 64) = v27;
      *(_BYTE *)(a2 + 76) = 0;
      *(_BYTE *)(a2 + 80) = 1;
    }
LABEL_9:
    v7 += 2;
    if ( (__int64 *)v8 == v7 )
      return;
  }
  if ( *(_BYTE *)(a2 + 56) )
  {
    v11 = (_BYTE *)v56;
    if ( !v57 || *(_QWORD *)(v56 + 8LL * v57 - 8) )
      goto LABEL_7;
    --v57;
LABEL_12:
    if ( !v55 || v57 > 1 || v60 && (v59 & 7) != 0 )
    {
      v25 = v56;
      goto LABEL_31;
    }
    HIWORD(v54) = sub_E91F10(v51, v55);
    LOBYTE(v54) = (v57 != 0) | v54 & 0xFE;
    v12 = 0;
    if ( v57 )
      v12 = (int)(2 * *(_QWORD *)(v56 + 8LL * v57 - 8)) >> 1;
    LODWORD(v54) = v54 & 1 | (2 * v12);
    if ( v60 )
      WORD2(v54) = 2 * (v59 >> 3) + 1;
    else
      WORD2(v54) = 0;
    v49 = sub_3211F40(a1, *v7 & 0xFFFFFFFFFFFFFFF8LL);
    v13 = v7[1];
    if ( v13 == -1 )
    {
      v16 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 400LL);
    }
    else
    {
      v14 = *(_QWORD *)(*(_QWORD *)a3 + 16 * v13);
      v15 = v14 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v14 & 4) != 0 )
        v16 = sub_3211FB0(a1, v15);
      else
        v16 = sub_3211F40(a1, v15);
    }
    v17 = sub_3202A70(a2 + 8, &v54);
    v20 = *(unsigned int *)(v17 + 8);
    if ( (_DWORD)v20 )
    {
      v21 = *(__int64 **)v17;
      v18 = v49;
      v22 = *(_QWORD *)v17 + 16 * v20 - 16;
      if ( v49 == *(_QWORD *)(v22 + 8) )
      {
        *(_QWORD *)(v22 + 8) = v16;
        goto LABEL_28;
      }
      v23 = *(unsigned int *)(v17 + 12);
      v24 = &v21[2 * v20];
      if ( v20 < v23 )
      {
LABEL_26:
        v24[1] = v16;
        *v24 = v49;
        LODWORD(v20) = *(_DWORD *)(v17 + 8);
LABEL_27:
        *(_DWORD *)(v17 + 8) = v20 + 1;
        goto LABEL_28;
      }
      v38 = v20 + 1;
      if ( v20 + 1 <= v23 )
      {
LABEL_70:
        v39 = &v21[2 * v20];
        *v39 = v49;
        v39[1] = v16;
        ++*(_DWORD *)(v17 + 8);
LABEL_28:
        if ( v61 )
          goto LABEL_29;
        goto LABEL_9;
      }
    }
    else
    {
      if ( *(_DWORD *)(v17 + 12) )
      {
        v24 = *(__int64 **)v17;
        if ( !*(_QWORD *)v17 )
          goto LABEL_27;
        goto LABEL_26;
      }
      v38 = 1;
    }
    sub_C8D5F0(v17, (const void *)(v17 + 16), v38, 0x10u, v18, v19);
    v20 = *(unsigned int *)(v17 + 8);
    v21 = *(__int64 **)v17;
    goto LABEL_70;
  }
  if ( v57 != 2 )
    goto LABEL_12;
  v25 = v56;
  if ( *(_QWORD *)(v56 + 8) )
  {
LABEL_31:
    v11 = (_BYTE *)v25;
LABEL_7:
    if ( v11 != v58 )
      _libc_free((unsigned __int64)v11);
    goto LABEL_9;
  }
  v28 = *(_DWORD *)(a2 + 24);
  ++*(_QWORD *)(a2 + 8);
  *(_BYTE *)(a2 + 56) = 1;
  if ( v28 )
  {
    v29 = 4 * v28;
    v30 = *(unsigned int *)(a2 + 32);
    if ( (unsigned int)(4 * v28) < 0x40 )
      v29 = 64;
    if ( (unsigned int)v30 > v29 )
    {
      v40 = v28 - 1;
      if ( v40 )
      {
        _BitScanReverse(&v40, v40);
        v41 = *(_QWORD **)(a2 + 16);
        v42 = 1 << (33 - (v40 ^ 0x1F));
        if ( v42 < 64 )
          v42 = 64;
        if ( (_DWORD)v30 == v42 )
        {
          *(_QWORD *)(a2 + 24) = 0;
          v48 = (_QWORD *)((char *)v41 + 12 * v30);
          do
          {
            if ( v41 )
              *v41 = -1;
            v41 = (_QWORD *)((char *)v41 + 12);
          }
          while ( v48 != v41 );
          goto LABEL_45;
        }
      }
      else
      {
        v41 = *(_QWORD **)(a2 + 16);
        v42 = 64;
      }
      sub_C7D6A0((__int64)v41, 12 * v30, 4);
      v43 = ((((((((4 * v42 / 3u + 1) | ((unsigned __int64)(4 * v42 / 3u + 1) >> 1)) >> 2)
               | (4 * v42 / 3u + 1)
               | ((unsigned __int64)(4 * v42 / 3u + 1) >> 1)) >> 4)
             | (((4 * v42 / 3u + 1) | ((unsigned __int64)(4 * v42 / 3u + 1) >> 1)) >> 2)
             | (4 * v42 / 3u + 1)
             | ((unsigned __int64)(4 * v42 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v42 / 3u + 1) | ((unsigned __int64)(4 * v42 / 3u + 1) >> 1)) >> 2)
             | (4 * v42 / 3u + 1)
             | ((unsigned __int64)(4 * v42 / 3u + 1) >> 1)) >> 4)
           | (((4 * v42 / 3u + 1) | ((unsigned __int64)(4 * v42 / 3u + 1) >> 1)) >> 2)
           | (4 * v42 / 3u + 1)
           | ((unsigned __int64)(4 * v42 / 3u + 1) >> 1)) >> 16;
      v44 = (v43
           | (((((((4 * v42 / 3u + 1) | ((unsigned __int64)(4 * v42 / 3u + 1) >> 1)) >> 2)
               | (4 * v42 / 3u + 1)
               | ((unsigned __int64)(4 * v42 / 3u + 1) >> 1)) >> 4)
             | (((4 * v42 / 3u + 1) | ((unsigned __int64)(4 * v42 / 3u + 1) >> 1)) >> 2)
             | (4 * v42 / 3u + 1)
             | ((unsigned __int64)(4 * v42 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v42 / 3u + 1) | ((unsigned __int64)(4 * v42 / 3u + 1) >> 1)) >> 2)
             | (4 * v42 / 3u + 1)
             | ((unsigned __int64)(4 * v42 / 3u + 1) >> 1)) >> 4)
           | (((4 * v42 / 3u + 1) | ((unsigned __int64)(4 * v42 / 3u + 1) >> 1)) >> 2)
           | (4 * v42 / 3u + 1)
           | ((unsigned __int64)(4 * v42 / 3u + 1) >> 1))
          + 1;
      *(_DWORD *)(a2 + 32) = v44;
      v45 = (_QWORD *)sub_C7D670(12 * v44, 4);
      v46 = *(unsigned int *)(a2 + 32);
      *(_QWORD *)(a2 + 24) = 0;
      *(_QWORD *)(a2 + 16) = v45;
      for ( i = (_QWORD *)((char *)v45 + 12 * v46); i != v45; v45 = (_QWORD *)((char *)v45 + 12) )
      {
        if ( v45 )
          *v45 = -1;
      }
    }
    else
    {
LABEL_42:
      v31 = *(_QWORD **)(a2 + 16);
      for ( j = (_QWORD *)((char *)v31 + 12 * v30); j != v31; v31 = (_QWORD *)((char *)v31 + 12) )
        *v31 = -1;
      *(_QWORD *)(a2 + 24) = 0;
    }
  }
  else if ( *(_DWORD *)(a2 + 28) )
  {
    v30 = *(unsigned int *)(a2 + 32);
    if ( (unsigned int)v30 <= 0x40 )
      goto LABEL_42;
    sub_C7D6A0(*(_QWORD *)(a2 + 16), 12 * v30, 4);
    *(_QWORD *)(a2 + 16) = 0;
    *(_QWORD *)(a2 + 24) = 0;
    *(_DWORD *)(a2 + 32) = 0;
  }
LABEL_45:
  v33 = *(_QWORD *)(a2 + 40);
  v34 = v33 + 40LL * *(unsigned int *)(a2 + 48);
  while ( v33 != v34 )
  {
    while ( 1 )
    {
      v34 -= 40;
      v35 = *(_QWORD *)(v34 + 8);
      if ( v35 == v34 + 24 )
        break;
      _libc_free(v35);
      if ( v33 == v34 )
        goto LABEL_49;
    }
  }
LABEL_49:
  *(_DWORD *)(a2 + 48) = 0;
  sub_3202DE0(a1, a2, a3);
  if ( v61 && (_BYTE *)v56 != v58 )
    _libc_free(v56);
}
