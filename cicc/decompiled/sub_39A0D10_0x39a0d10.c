// Function: sub_39A0D10
// Address: 0x39a0d10
//
__int64 __fastcall sub_39A0D10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rdi
  unsigned int v7; // esi
  __int64 v8; // r9
  __int64 v9; // r8
  unsigned int v10; // ecx
  __int64 v11; // r12
  __int64 v12; // rax
  unsigned int v13; // ebx
  __int64 v14; // rdx
  _QWORD *v15; // r15
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rcx
  unsigned __int64 v20; // r13
  __int64 v21; // rcx
  __int64 v22; // rax
  _QWORD *v23; // r8
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  char v27; // di
  __int64 v29; // rax
  __int64 *v30; // rax
  int v31; // r10d
  __int64 v32; // rdx
  int v33; // eax
  int v34; // edx
  int v35; // eax
  int v36; // ecx
  __int64 v37; // rdi
  unsigned int v38; // eax
  __int64 v39; // rsi
  int v40; // r9d
  __int64 v41; // r8
  int v42; // eax
  int v43; // eax
  __int64 v44; // rsi
  int v45; // r8d
  unsigned int v46; // r15d
  __int64 v47; // rdi
  __int64 v48; // rcx
  _QWORD *v49; // [rsp+0h] [rbp-40h]
  __int64 v50; // [rsp+8h] [rbp-38h]
  unsigned __int64 v51; // [rsp+8h] [rbp-38h]

  v6 = a1 + 264;
  v7 = *(_DWORD *)(a1 + 288);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 264);
    goto LABEL_42;
  }
  v8 = v7 - 1;
  v9 = *(_QWORD *)(a1 + 272);
  v10 = v8 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v11 = v9 + 136LL * v10;
  v12 = *(_QWORD *)v11;
  if ( *(_QWORD *)v11 != a2 )
  {
    v31 = 1;
    v32 = 0;
    while ( v12 != -8 )
    {
      if ( !v32 && v12 == -16 )
        v32 = v11;
      v10 = v8 & (v31 + v10);
      v11 = v9 + 136LL * v10;
      v12 = *(_QWORD *)v11;
      if ( *(_QWORD *)v11 == a2 )
        goto LABEL_3;
      ++v31;
    }
    v33 = *(_DWORD *)(a1 + 280);
    if ( v32 )
      v11 = v32;
    ++*(_QWORD *)(a1 + 264);
    v34 = v33 + 1;
    if ( 4 * (v33 + 1) < 3 * v7 )
    {
      if ( v7 - *(_DWORD *)(a1 + 284) - v34 > v7 >> 3 )
        goto LABEL_35;
      sub_39A0940(v6, v7);
      v42 = *(_DWORD *)(a1 + 288);
      if ( v42 )
      {
        v43 = v42 - 1;
        v44 = *(_QWORD *)(a1 + 272);
        v45 = 1;
        v46 = v43 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v11 = v44 + 136LL * v46;
        v34 = *(_DWORD *)(a1 + 280) + 1;
        v47 = 0;
        v48 = *(_QWORD *)v11;
        if ( *(_QWORD *)v11 != a2 )
        {
          while ( v48 != -8 )
          {
            if ( v48 == -16 && !v47 )
              v47 = v11;
            v46 = v43 & (v45 + v46);
            v11 = v44 + 136LL * v46;
            v48 = *(_QWORD *)v11;
            if ( *(_QWORD *)v11 == a2 )
              goto LABEL_35;
            ++v45;
          }
          if ( v47 )
            v11 = v47;
        }
        goto LABEL_35;
      }
      goto LABEL_72;
    }
LABEL_42:
    sub_39A0940(v6, 2 * v7);
    v35 = *(_DWORD *)(a1 + 288);
    if ( v35 )
    {
      v36 = v35 - 1;
      v37 = *(_QWORD *)(a1 + 272);
      v38 = (v35 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v11 = v37 + 136LL * v38;
      v39 = *(_QWORD *)v11;
      v34 = *(_DWORD *)(a1 + 280) + 1;
      if ( *(_QWORD *)v11 != a2 )
      {
        v40 = 1;
        v41 = 0;
        while ( v39 != -8 )
        {
          if ( !v41 && v39 == -16 )
            v41 = v11;
          v38 = v36 & (v40 + v38);
          v11 = v37 + 136LL * v38;
          v39 = *(_QWORD *)v11;
          if ( *(_QWORD *)v11 == a2 )
            goto LABEL_35;
          ++v40;
        }
        if ( v41 )
          v11 = v41;
      }
LABEL_35:
      *(_DWORD *)(a1 + 280) = v34;
      if ( *(_QWORD *)v11 != -8 )
        --*(_DWORD *)(a1 + 284);
      v23 = (_QWORD *)(v11 + 8);
      *(_QWORD *)v11 = a2;
      v20 = v11 + 16;
      memset((void *)(v11 + 8), 0, 0x80u);
      v30 = (__int64 *)(v11 + 72);
      *(_QWORD *)(v11 + 32) = v11 + 16;
      *(_QWORD *)(v11 + 40) = v11 + 16;
      *(_QWORD *)(v11 + 56) = v11 + 72;
      *(_QWORD *)(v11 + 64) = 0x800000000LL;
      v13 = *(unsigned __int16 *)(*(_QWORD *)a3 + 32LL);
      if ( *(_WORD *)(*(_QWORD *)a3 + 32LL) )
        goto LABEL_38;
LABEL_28:
      *v30 = a3;
      ++*(_DWORD *)(v11 + 64);
      return 1;
    }
LABEL_72:
    ++*(_DWORD *)(a1 + 280);
    BUG();
  }
LABEL_3:
  v13 = *(unsigned __int16 *)(*(_QWORD *)a3 + 32LL);
  if ( !*(_WORD *)(*(_QWORD *)a3 + 32LL) )
  {
    v29 = *(unsigned int *)(v11 + 64);
    if ( (unsigned int)v29 >= *(_DWORD *)(v11 + 68) )
    {
      sub_16CD150(v11 + 56, (const void *)(v11 + 72), 0, 8, v9, v8);
      v30 = (__int64 *)(*(_QWORD *)(v11 + 56) + 8LL * *(unsigned int *)(v11 + 64));
    }
    else
    {
      v30 = (__int64 *)(*(_QWORD *)(v11 + 56) + 8 * v29);
    }
    goto LABEL_28;
  }
  v14 = *(_QWORD *)(v11 + 24);
  v15 = (_QWORD *)(v11 + 16);
  if ( !v14 )
  {
    v20 = v11 + 16;
    v23 = (_QWORD *)(v11 + 8);
LABEL_38:
    v15 = (_QWORD *)v20;
    goto LABEL_18;
  }
  v16 = v11 + 16;
  v17 = *(_QWORD *)(v11 + 24);
  do
  {
    while ( 1 )
    {
      v18 = *(_QWORD *)(v17 + 16);
      v19 = *(_QWORD *)(v17 + 24);
      if ( v13 <= *(_DWORD *)(v17 + 32) )
        break;
      v17 = *(_QWORD *)(v17 + 24);
      if ( !v19 )
        goto LABEL_9;
    }
    v16 = v17;
    v17 = *(_QWORD *)(v17 + 16);
  }
  while ( v18 );
LABEL_9:
  if ( v15 == (_QWORD *)v16 || v13 < *(_DWORD *)(v16 + 32) )
  {
    v20 = v11 + 16;
    do
    {
      while ( 1 )
      {
        v21 = *(_QWORD *)(v14 + 16);
        v22 = *(_QWORD *)(v14 + 24);
        if ( v13 <= *(_DWORD *)(v14 + 32) )
          break;
        v14 = *(_QWORD *)(v14 + 24);
        if ( !v22 )
          goto LABEL_15;
      }
      v20 = v14;
      v14 = *(_QWORD *)(v14 + 16);
    }
    while ( v21 );
LABEL_15:
    if ( (_QWORD *)v20 != v15 && v13 >= *(_DWORD *)(v20 + 32) )
      goto LABEL_23;
    v23 = (_QWORD *)(v11 + 8);
LABEL_18:
    v49 = v23;
    v50 = v20;
    v24 = sub_22077B0(0x30u);
    *(_DWORD *)(v24 + 32) = v13;
    v20 = v24;
    *(_QWORD *)(v24 + 40) = 0;
    v25 = sub_39A0840(v49, v50, (unsigned int *)(v24 + 32));
    if ( v26 )
    {
      v27 = v15 == (_QWORD *)v26 || v25 || v13 < *(_DWORD *)(v26 + 32);
      sub_220F040(v27, v20, (_QWORD *)v26, v15);
      ++*(_QWORD *)(v11 + 48);
    }
    else
    {
      v51 = v25;
      j_j___libc_free_0(v20);
      v20 = v51;
    }
LABEL_23:
    *(_QWORD *)(v20 + 40) = a3;
    return 1;
  }
  sub_3988A40(*(_QWORD *)(v16 + 40), a3, v14, v19, v9, v8);
  return 0;
}
