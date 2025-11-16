// Function: sub_1625C10
// Address: 0x1625c10
//
void __fastcall sub_1625C10(__int64 a1, int a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v6; // rsi
  unsigned __int8 *v7; // rsi
  __int64 v8; // rax
  __int64 v9; // r14
  unsigned int v10; // esi
  __int64 v11; // r15
  __int64 v12; // rcx
  __int64 v13; // r9
  unsigned int v14; // eax
  __int64 *v15; // rdi
  __int64 v16; // r8
  int v17; // eax
  unsigned int *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // r15
  unsigned int v21; // esi
  __int64 v22; // rdi
  __int64 v23; // rcx
  __int64 v24; // r8
  unsigned int v25; // eax
  _QWORD *v26; // r14
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // r13
  int v30; // eax
  int v31; // edx
  __int64 v32; // rdi
  unsigned int v33; // eax
  int v34; // esi
  __int64 *v35; // r14
  __int64 v36; // rcx
  __int64 v37; // r15
  unsigned __int64 v38; // r12
  __int64 v39; // rsi
  __int64 *v40; // rdx
  int v41; // edi
  int v42; // r11d
  int v43; // eax
  int v44; // r10d
  int v45; // eax
  int v46; // edx
  __int64 v47; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v48[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = a3;
  if ( a3 )
  {
    if ( !a2 )
    {
LABEL_3:
      sub_15C7080(v48, a3);
      v6 = *(_QWORD *)(a1 + 48);
      if ( v6 )
        sub_161E7C0(a1 + 48, v6);
      v7 = (unsigned __int8 *)v48[0];
      *(_QWORD *)(a1 + 48) = v48[0];
      if ( v7 )
        sub_1623210((__int64)v48, v7, a1 + 48);
      return;
    }
    v47 = a1;
    v8 = sub_16498A0(a1);
    v9 = *(_QWORD *)v8;
    v10 = *(_DWORD *)(*(_QWORD *)v8 + 2728LL);
    v11 = *(_QWORD *)v8 + 2704LL;
    if ( v10 )
    {
      v12 = v47;
      v13 = *(_QWORD *)(v9 + 2712);
      v14 = (v10 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
      v15 = (__int64 *)(v13 + 56LL * v14);
      v16 = *v15;
      if ( v47 == *v15 )
      {
LABEL_9:
        v17 = *((_DWORD *)v15 + 4);
        v18 = (unsigned int *)(v15 + 1);
        if ( v17 )
        {
LABEL_10:
          sub_1623F80(v18, a2, v4);
          return;
        }
LABEL_37:
        *(_WORD *)(a1 + 18) |= 0x8000u;
        goto LABEL_10;
      }
      v42 = 1;
      v40 = 0;
      while ( v16 != -8 )
      {
        if ( v40 || v16 != -16 )
          v15 = v40;
        v14 = (v10 - 1) & (v42 + v14);
        v16 = *(_QWORD *)(v13 + 56LL * v14);
        if ( v47 == v16 )
        {
          v15 = (__int64 *)(v13 + 56LL * v14);
          goto LABEL_9;
        }
        ++v42;
        v40 = v15;
        v15 = (__int64 *)(v13 + 56LL * v14);
      }
      v43 = *(_DWORD *)(v9 + 2720);
      if ( !v40 )
        v40 = v15;
      ++*(_QWORD *)(v9 + 2704);
      v41 = v43 + 1;
      if ( 4 * (v43 + 1) < 3 * v10 )
      {
        if ( v10 - *(_DWORD *)(v9 + 2724) - v41 > v10 >> 3 )
          goto LABEL_34;
LABEL_33:
        sub_1624590(v11, v10);
        sub_1621460(v11, &v47, v48);
        v40 = (__int64 *)v48[0];
        v12 = v47;
        v41 = *(_DWORD *)(v9 + 2720) + 1;
LABEL_34:
        *(_DWORD *)(v9 + 2720) = v41;
        if ( *v40 != -8 )
          --*(_DWORD *)(v9 + 2724);
        *v40 = v12;
        v18 = (unsigned int *)(v40 + 1);
        v40[1] = (__int64)(v40 + 3);
        v40[2] = 0x200000000LL;
        goto LABEL_37;
      }
    }
    else
    {
      ++*(_QWORD *)(v9 + 2704);
    }
    v10 *= 2;
    goto LABEL_33;
  }
  if ( *(_QWORD *)(a1 + 48) )
  {
    if ( !a2 )
      goto LABEL_3;
    if ( *(__int16 *)(a1 + 18) >= 0 )
      return;
  }
  else
  {
    if ( *(__int16 *)(a1 + 18) >= 0 )
      return;
    if ( !a2 )
      goto LABEL_3;
  }
  v47 = a1;
  v19 = sub_16498A0(a1);
  v20 = *(_QWORD *)v19;
  v21 = *(_DWORD *)(*(_QWORD *)v19 + 2728LL);
  v22 = *(_QWORD *)v19 + 2704LL;
  if ( !v21 )
  {
    ++*(_QWORD *)(v20 + 2704);
LABEL_58:
    v21 *= 2;
    goto LABEL_59;
  }
  v23 = v47;
  v24 = *(_QWORD *)(v20 + 2712);
  v25 = (v21 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
  v26 = (_QWORD *)(v24 + 56LL * v25);
  v27 = *v26;
  if ( v47 == *v26 )
    goto LABEL_19;
  v44 = 1;
  while ( v27 != -8 )
  {
    if ( !v4 && v27 == -16 )
      v4 = (__int64)v26;
    v25 = (v21 - 1) & (v44 + v25);
    v26 = (_QWORD *)(v24 + 56LL * v25);
    v27 = *v26;
    if ( v47 == *v26 )
      goto LABEL_19;
    ++v44;
  }
  v45 = *(_DWORD *)(v20 + 2720);
  if ( v4 )
    v26 = (_QWORD *)v4;
  ++*(_QWORD *)(v20 + 2704);
  v46 = v45 + 1;
  if ( 4 * (v45 + 1) >= 3 * v21 )
    goto LABEL_58;
  if ( v21 - *(_DWORD *)(v20 + 2724) - v46 <= v21 >> 3 )
  {
LABEL_59:
    sub_1624590(v22, v21);
    sub_1621460(v22, &v47, v48);
    v26 = (_QWORD *)v48[0];
    v23 = v47;
    v46 = *(_DWORD *)(v20 + 2720) + 1;
  }
  *(_DWORD *)(v20 + 2720) = v46;
  if ( *v26 != -8 )
    --*(_DWORD *)(v20 + 2724);
  *v26 = v23;
  v26[1] = v26 + 3;
  v26[2] = 0x200000000LL;
LABEL_19:
  sub_16235A0(v26 + 1, a2);
  if ( !*((_DWORD *)v26 + 4) )
  {
    v28 = sub_16498A0(a1);
    v29 = *(_QWORD *)v28;
    v30 = *(_DWORD *)(*(_QWORD *)v28 + 2728LL);
    if ( v30 )
    {
      v31 = v30 - 1;
      v32 = *(_QWORD *)(v29 + 2712);
      v33 = (v30 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v34 = 1;
      v35 = (__int64 *)(v32 + 56LL * v33);
      v36 = *v35;
      if ( a1 == *v35 )
      {
LABEL_22:
        v37 = v35[1];
        v38 = v37 + 16LL * *((unsigned int *)v35 + 4);
        if ( v37 != v38 )
        {
          do
          {
            v39 = *(_QWORD *)(v38 - 8);
            v38 -= 16LL;
            if ( v39 )
              sub_161E7C0(v38 + 8, v39);
          }
          while ( v37 != v38 );
          v38 = v35[1];
        }
        if ( (__int64 *)v38 != v35 + 3 )
          _libc_free(v38);
        *v35 = -16;
        --*(_DWORD *)(v29 + 2720);
        ++*(_DWORD *)(v29 + 2724);
      }
      else
      {
        while ( v36 != -8 )
        {
          v33 = v31 & (v34 + v33);
          v35 = (__int64 *)(v32 + 56LL * v33);
          v36 = *v35;
          if ( a1 == *v35 )
            goto LABEL_22;
          ++v34;
        }
      }
    }
    *(_WORD *)(a1 + 18) &= ~0x8000u;
  }
}
