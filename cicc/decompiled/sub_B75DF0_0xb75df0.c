// Function: sub_B75DF0
// Address: 0xb75df0
//
void __fastcall sub_B75DF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v6; // ebx
  __int64 v7; // rbx
  __int64 v8; // r15
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r12
  int *v13; // rbx
  int *v14; // r15
  __int64 v15; // r9
  __int64 *v16; // r8
  int v17; // eax
  __int64 v18; // rcx
  int *v19; // rdi
  char v20; // al
  unsigned int v21; // eax
  __int64 v22; // r12
  int v23; // r15d
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  unsigned int v27; // ebx
  unsigned int v28; // eax
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // rax
  _QWORD *v31; // rax
  char v32; // al
  _QWORD *m; // rbx
  _QWORD *k; // rbx
  _QWORD *i; // rbx
  _QWORD *j; // r15
  __int64 *v37; // [rsp+8h] [rbp-A8h]
  __int64 v38; // [rsp+10h] [rbp-A0h]
  __int64 *v39; // [rsp+10h] [rbp-A0h]
  __int64 *v40; // [rsp+10h] [rbp-A0h]
  __int64 v41; // [rsp+20h] [rbp-90h] BYREF
  _QWORD *v42; // [rsp+28h] [rbp-88h]
  int v43; // [rsp+40h] [rbp-70h]
  char v44; // [rsp+44h] [rbp-6Ch]
  __int64 v45; // [rsp+48h] [rbp-68h] BYREF
  _QWORD *v46; // [rsp+50h] [rbp-60h]
  __int64 v47; // [rsp+60h] [rbp-50h] BYREF
  _QWORD *v48; // [rsp+68h] [rbp-48h] BYREF
  _QWORD *v49; // [rsp+70h] [rbp-40h]

  v6 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( v6 )
  {
    v21 = 4 * v6;
    v22 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)(4 * v6) < 0x40 )
      v21 = 64;
    if ( (unsigned int)v22 <= v21 )
      goto LABEL_4;
    v23 = 64;
    sub_B74660(a1, a2, a3, a4, a5);
    v27 = v6 - 1;
    if ( v27 )
    {
      _BitScanReverse(&v28, v27);
      v28 ^= 0x1Fu;
      v25 = 33 - v28;
      v23 = 1 << (33 - v28);
      if ( v23 < 64 )
        v23 = 64;
    }
    if ( *(_DWORD *)(a1 + 24) != v23 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 8), 40 * v22, 8);
      a2 = 8;
      v29 = ((((((((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
               | (4 * v23 / 3u + 1)
               | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 4)
             | (((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
             | (4 * v23 / 3u + 1)
             | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
             | (4 * v23 / 3u + 1)
             | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 4)
           | (((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
           | (4 * v23 / 3u + 1)
           | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 16;
      v30 = (v29
           | (((((((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
               | (4 * v23 / 3u + 1)
               | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 4)
             | (((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
             | (4 * v23 / 3u + 1)
             | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
             | (4 * v23 / 3u + 1)
             | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 4)
           | (((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
           | (4 * v23 / 3u + 1)
           | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 24) = v30;
      *(_QWORD *)(a1 + 8) = sub_C7D670(40 * v30, 8);
    }
LABEL_44:
    sub_B75C10(a1, a2, v24, v25, v26);
    return;
  }
  a3 = *(unsigned int *)(a1 + 20);
  if ( !(_DWORD)a3 )
    return;
  v7 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)v7 > 0x40 )
  {
    sub_B74660(a1, a2, a3, a4, a5);
    if ( *(_DWORD *)(a1 + 24) )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 8), 40 * v7, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return;
    }
    goto LABEL_44;
  }
LABEL_4:
  v8 = sub_C33690(a1, a2, a3, a4, a5);
  v12 = sub_C33340(a1, a2, v9, v10, v11);
  if ( v8 == v12 )
    sub_C3C5A0(&v47, v8, 1);
  else
    sub_C36740(&v47, v8, 1);
  v43 = -1;
  v44 = 1;
  if ( v47 == v12 )
  {
    sub_C3C840(&v45, &v47);
    if ( v47 != v12 )
      goto LABEL_8;
  }
  else
  {
    sub_C338E0(&v45, &v47);
    if ( v47 != v12 )
    {
LABEL_8:
      sub_C338F0(&v47);
      goto LABEL_9;
    }
  }
  if ( v48 )
  {
    for ( i = &v48[3 * *(v48 - 1)]; v48 != i; sub_91D830(i) )
      i -= 3;
    j_j_j___libc_free_0_0(i - 1);
  }
LABEL_9:
  if ( v8 == v12 )
    sub_C3C5A0(&v41, v12, 2);
  else
    sub_C36740(&v41, v8, 2);
  BYTE4(v47) = 0;
  LODWORD(v47) = -2;
  if ( v12 == v41 )
    sub_C3C840(&v48, &v41);
  else
    sub_C338E0(&v48, &v41);
  if ( v12 == v41 )
  {
    if ( v42 )
    {
      for ( j = &v42[3 * *(v42 - 1)]; v42 != j; sub_91D830(j) )
        j -= 3;
      j_j_j___libc_free_0_0(j - 1);
    }
  }
  else
  {
    sub_C338F0(&v41);
  }
  v13 = *(int **)(a1 + 8);
  v14 = &v13[10 * *(unsigned int *)(a1 + 24)];
  while ( v14 != v13 )
  {
    v17 = *v13;
    v16 = (__int64 *)(v13 + 2);
    if ( *v13 == v43 && *((_BYTE *)v13 + 4) == v44 )
    {
      v18 = *((_QWORD *)v13 + 1);
      if ( v18 == v45 )
      {
        v19 = v13 + 2;
        if ( v12 == v18 )
          v20 = sub_C3E590(v19);
        else
          v20 = sub_C33D00(v19);
        v16 = (__int64 *)(v13 + 2);
        if ( v20 )
          goto LABEL_23;
        v17 = *v13;
      }
    }
    if ( v17 != (_DWORD)v47
      || *((_BYTE *)v13 + 4) != BYTE4(v47)
      || (v31 = (_QWORD *)*((_QWORD *)v13 + 1), v31 != v48)
      || ((v40 = v16, (_QWORD *)v12 == v31) ? (v32 = sub_C3E590(v16)) : (v32 = sub_C33D00(v16)), v16 = v40, !v32) )
    {
      v15 = *((_QWORD *)v13 + 4);
      if ( v15 )
      {
        v37 = v16;
        v38 = *((_QWORD *)v13 + 4);
        sub_91D830((_QWORD *)(v15 + 24));
        sub_BD7260(v38);
        sub_BD2DD0(v38);
        v16 = v37;
      }
    }
    *v13 = v43;
    *((_BYTE *)v13 + 4) = v44;
    if ( v12 == *((_QWORD *)v13 + 1) )
    {
      if ( v12 != v45 )
      {
LABEL_46:
        if ( v16 != &v45 )
        {
          v39 = v16;
          sub_91D830(v16);
          if ( v12 == v45 )
            sub_C3C790(v39, &v45);
          else
            sub_C33EB0(v39, &v45);
        }
        goto LABEL_23;
      }
      sub_C3C9E0(v16, &v45);
    }
    else
    {
      if ( v12 == v45 )
        goto LABEL_46;
      sub_C33E70(v16, &v45);
    }
LABEL_23:
    v13 += 10;
  }
  if ( (_QWORD *)v12 == v48 )
  {
    if ( v49 )
    {
      for ( k = &v49[3 * *(v49 - 1)]; v49 != k; sub_91D830(k) )
        k -= 3;
      j_j_j___libc_free_0_0(k - 1);
    }
  }
  else
  {
    sub_C338F0(&v48);
  }
  *(_QWORD *)(a1 + 16) = 0;
  if ( v12 == v45 )
  {
    if ( v46 )
    {
      for ( m = &v46[3 * *(v46 - 1)]; v46 != m; sub_91D830(m) )
        m -= 3;
      j_j_j___libc_free_0_0(m - 1);
    }
  }
  else
  {
    sub_C338F0(&v45);
  }
}
