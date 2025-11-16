// Function: sub_1ED7320
// Address: 0x1ed7320
//
__int64 __fastcall sub_1ED7320(__int64 *a1, unsigned __int64 a2)
{
  __int64 (*v3)(void); // rdx
  __int64 v4; // rax
  __int64 (*v5)(); // rax
  __int64 v6; // rax
  unsigned int v7; // r13d
  __int64 v8; // r12
  _QWORD *v9; // rax
  _QWORD *v10; // rcx
  _DWORD *v11; // rax
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // rdi
  unsigned __int64 v16; // rcx
  __int64 v17; // r8
  _WORD *v18; // r9
  unsigned __int16 *v19; // r12
  unsigned __int64 v20; // r13
  unsigned __int64 v21; // rax
  __int64 *v22; // rdx
  _QWORD *v23; // r8
  __int64 v24; // r11
  __int64 v25; // r10
  int v26; // r14d
  _WORD *v27; // rdi
  int v28; // edx
  unsigned __int16 *v29; // r13
  unsigned __int16 *v30; // rax
  int v31; // edi
  __int64 v32; // rax
  unsigned __int16 *v33; // rax
  int v34; // r10d
  int v35; // edx
  __int64 v36; // rax
  __int64 v37; // r14
  unsigned int v38; // edi
  __int64 result; // rax
  unsigned __int64 v40; // rax
  unsigned int v41; // r13d
  __int64 v42; // r12
  size_t v43; // r12
  __int64 v44; // r8
  void *v45; // rdi
  __int64 v46; // rax
  __int64 v47; // rdx
  void *v48; // rdi
  void *v49; // r13
  int v50; // eax
  int v51; // edx
  __int64 v52; // rax
  __int64 v53; // rax

  a1[2] = a2;
  v3 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 112LL);
  v4 = 0;
  if ( v3 != sub_1D00B10 )
  {
    v4 = v3();
    a2 = a1[2];
  }
  if ( v4 == a1[3] )
  {
    v46 = sub_1E6A620(*(_QWORD **)(a2 + 40));
    v17 = 0;
    v19 = (unsigned __int16 *)v46;
    if ( a1[4] == v46 )
      goto LABEL_35;
  }
  else
  {
    v5 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 112LL);
    if ( v5 == sub_1D00B10 )
    {
      a1[3] = 0;
      BUG();
    }
    v6 = v5();
    a1[3] = v6;
    v7 = (__int64)(*(_QWORD *)(v6 + 264) - *(_QWORD *)(v6 + 256)) >> 3;
    v8 = 3LL * v7;
    v9 = (_QWORD *)sub_2207820(v8 * 8 + 8);
    if ( v9 )
    {
      *v9 = v7;
      v10 = v9 + 1;
      if ( v7 )
      {
        v11 = v9 + 1;
        do
        {
          *v11 = 0;
          v11 += 6;
          *(v11 - 5) = 0;
          *((_BYTE *)v11 - 16) = 0;
          *((_BYTE *)v11 - 15) = 0;
          *((_WORD *)v11 - 7) = 0;
          *((_QWORD *)v11 - 1) = 0;
        }
        while ( &v10[v8] != (_QWORD *)v11 );
      }
    }
    else
    {
      v10 = 0;
    }
    v12 = *a1;
    *a1 = (__int64)v10;
    if ( v12 )
    {
      v13 = 24LL * *(_QWORD *)(v12 - 8);
      v14 = v12 + v13;
      if ( v12 != v12 + v13 )
      {
        do
        {
          v15 = *(_QWORD *)(v14 - 8);
          v14 -= 24;
          if ( v15 )
            j_j___libc_free_0_0(v15);
        }
        while ( v12 != v14 );
        v13 = 24LL * *(_QWORD *)(v12 - 8);
      }
      a2 = v13 + 8;
      j_j_j___libc_free_0_0(v12 - 8);
    }
    v19 = (unsigned __int16 *)sub_1E6A620(*(_QWORD **)(a1[2] + 40));
  }
  v20 = *(unsigned int *)(a1[3] + 16);
  v21 = *((unsigned int *)a1 + 12);
  if ( v20 >= v21 )
  {
    if ( v20 <= v21 )
      goto LABEL_19;
    if ( v20 > *((unsigned int *)a1 + 13) )
    {
      a2 = (unsigned __int64)(a1 + 7);
      sub_16CD150((__int64)(a1 + 5), a1 + 7, *(unsigned int *)(a1[3] + 16), 2, v17, (int)v18);
      v21 = *((unsigned int *)a1 + 12);
    }
    v47 = a1[5];
    v48 = (void *)(v47 + 2 * v21);
    if ( v48 != (void *)(v47 + 2 * v20) )
    {
      a2 = 0;
      memset(v48, 0, 2 * (v20 - v21));
    }
  }
  *((_DWORD *)a1 + 12) = v20;
LABEL_19:
  v22 = (__int64 *)*v19;
  if ( (_WORD)v22 )
  {
    v18 = v19;
    do
    {
      v23 = (_QWORD *)a1[3];
      if ( !v23 )
        BUG();
      v24 = v23[1];
      v25 = v23[7];
      v26 = 0;
      a2 = (unsigned int)v22 * (*(_DWORD *)(v24 + 24LL * (unsigned __int16)v22 + 16) & 0xF);
      v27 = (_WORD *)(v25 + 2LL * (*(_DWORD *)(v24 + 24LL * (unsigned __int16)v22 + 16) >> 4));
      v28 = 0;
      v16 = (unsigned __int64)(v27 + 1);
      LOWORD(a2) = *v27 + a2;
      while ( 1 )
      {
        v29 = (unsigned __int16 *)v16;
        if ( !v16 )
          break;
        while ( 1 )
        {
          v30 = (unsigned __int16 *)(v23[6] + 4LL * (unsigned __int16)a2);
          v31 = *v30;
          v28 = v30[1];
          if ( (_WORD)v31 )
          {
            while ( 1 )
            {
              v32 = v25 + 2LL * *(unsigned int *)(v24 + 24LL * (unsigned __int16)v31 + 8);
              if ( v32 )
                goto LABEL_28;
              if ( !(_WORD)v28 )
                break;
              v31 = v28;
              v28 = 0;
            }
            v26 = v31;
          }
          v50 = *v29;
          v16 = 0;
          ++v29;
          if ( !(_WORD)v50 )
            break;
          a2 = (unsigned int)(v50 + a2);
          v16 = (unsigned __int64)v29;
          if ( !v29 )
            goto LABEL_65;
        }
      }
LABEL_65:
      v31 = v26;
      v32 = 0;
LABEL_28:
      while ( v16 )
      {
        v32 += 2;
        *(_WORD *)(a1[5] + 2LL * (unsigned __int16)v31) = *v18;
        v34 = *(unsigned __int16 *)(v32 - 2);
        v31 += v34;
        if ( !(_WORD)v34 )
        {
          if ( (_WORD)v28 )
          {
            v32 = v23[7] + 2LL * *(unsigned int *)(v23[1] + 24LL * (unsigned __int16)v28 + 8);
            v31 = v28;
            v28 = 0;
          }
          else
          {
            v35 = *(unsigned __int16 *)v16;
            a2 = (unsigned int)(v35 + a2);
            if ( !(_WORD)v35 )
            {
              v16 = 0;
              break;
            }
            v16 += 2LL;
            v33 = (unsigned __int16 *)(v23[6] + 4LL * (unsigned __int16)a2);
            v31 = *v33;
            v28 = v33[1];
            v32 = v23[7] + 2LL * *(unsigned int *)(v23[1] + 24LL * (unsigned __int16)v31 + 8);
          }
        }
      }
      v22 = (__int64 *)(unsigned __int16)v18[1];
      ++v18;
    }
    while ( (_WORD)v22 );
  }
  v17 = 1;
LABEL_35:
  v36 = a1[2];
  a1[4] = (__int64)v19;
  v37 = *(_QWORD *)(v36 + 40);
  v38 = *(_DWORD *)(v37 + 320);
  if ( v38 == *((_DWORD *)a1 + 20) )
  {
    result = (v38 + 63) >> 6;
    if ( !((v38 + 63) >> 6) )
    {
LABEL_60:
      if ( !(_BYTE)v17 )
        return result;
      goto LABEL_44;
    }
    v22 = (__int64 *)(unsigned int)result;
    a2 = *(_QWORD *)(v37 + 304);
    v16 = a1[8];
    result = 0;
    while ( 1 )
    {
      v18 = *(_WORD **)(v16 + 8 * result);
      if ( *(_WORD **)(a2 + 8 * result) != v18 )
        break;
      if ( v22 == (__int64 *)++result )
        goto LABEL_60;
    }
  }
  v22 = a1 + 8;
  if ( a1 + 8 != (__int64 *)(v37 + 304) )
  {
    v40 = a1[9];
    v22 = (__int64 *)v38;
    *((_DWORD *)a1 + 20) = v38;
    v41 = (v38 + 63) >> 6;
    v42 = v41;
    v16 = v40 << 6;
    if ( v38 > v40 << 6 )
    {
      v49 = (void *)malloc(8LL * v41);
      if ( !v49 )
      {
        if ( 8 * v42 || (v53 = malloc(1u)) == 0 )
          sub_16BD1C0("Allocation failed", 1u);
        else
          v49 = (void *)v53;
      }
      a2 = *(_QWORD *)(v37 + 304);
      memcpy(v49, (const void *)a2, 8 * v42);
      _libc_free(a1[8]);
      a1[8] = (__int64)v49;
      a1[9] = v42;
      goto LABEL_44;
    }
    if ( v38 )
    {
      a2 = *(_QWORD *)(v37 + 304);
      memcpy((void *)a1[8], (const void *)a2, 8LL * v41);
      v51 = *((_DWORD *)a1 + 20);
      v40 = a1[9];
      v41 = (unsigned int)(v51 + 63) >> 6;
      v42 = v41;
      if ( v40 <= v41 )
      {
LABEL_70:
        v22 = (__int64 *)(v51 & 0x3F);
        if ( (_DWORD)v22 )
        {
          a2 = a1[8];
          v16 = (unsigned int)v22;
          *(_QWORD *)(a2 + 8LL * (v41 - 1)) &= ~(-1LL << (char)v22);
        }
        goto LABEL_44;
      }
    }
    else if ( v40 <= v41 )
    {
      goto LABEL_44;
    }
    v52 = v40 - v42;
    if ( v52 )
    {
      a2 = 0;
      memset((void *)(a1[8] + 8 * v42), 0, 8 * v52);
    }
    v51 = *((_DWORD *)a1 + 20);
    goto LABEL_70;
  }
LABEL_44:
  v43 = 4LL
      * (*(unsigned int (__fastcall **)(__int64, unsigned __int64, __int64 *, unsigned __int64, __int64, _WORD *))(*(_QWORD *)a1[3] + 200LL))(
          a1[3],
          a2,
          v22,
          v16,
          v17,
          v18);
  result = sub_2207820(v43);
  v44 = a1[11];
  v45 = (void *)result;
  a1[11] = result;
  if ( v44 )
  {
    result = j_j___libc_free_0_0(v44);
    v45 = (void *)a1[11];
  }
  if ( v43 )
    result = (__int64)memset(v45, 0, v43);
  ++*((_DWORD *)a1 + 2);
  return result;
}
