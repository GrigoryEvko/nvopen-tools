// Function: sub_2F7B910
// Address: 0x2f7b910
//
__int64 __fastcall sub_2F7B910(__int64 *a1, __int64 a2)
{
  __int64 v2; // rbx
  _QWORD *v3; // r13
  _DWORD *v4; // r15
  unsigned __int16 v5; // cx
  int v7; // eax
  __int64 v8; // r14
  unsigned int v9; // eax
  __int64 v10; // rdi
  __int64 (*v11)(); // rax
  __int64 v12; // rax
  _QWORD *v13; // r14
  __int64 v14; // r8
  unsigned __int16 *v15; // r9
  _QWORD *v16; // rax
  __int64 v17; // rdx
  unsigned __int64 v18; // rcx
  __int64 v19; // rsi
  __int64 v20; // rdx
  unsigned __int16 *v21; // rax
  int v22; // edi
  __int16 *v23; // rax
  unsigned int v24; // r10d
  unsigned __int64 i; // rcx
  __int64 (*v26)(); // rax
  _QWORD *v27; // r10
  __int64 v28; // r14
  int v29; // r15d
  unsigned int v30; // r12d
  __int64 v31; // rax
  __int64 v32; // rdx
  char *v33; // rax
  char *v34; // r9
  unsigned int v35; // edi
  __int16 v36; // cx
  __int64 (*v37)(); // rax
  __int64 v38; // rdi
  __int64 (*v39)(); // rax
  __int64 v40; // rdx
  __int64 v41; // r8
  unsigned __int16 *v42; // rbx
  unsigned __int16 *v43; // r12
  char *v44; // rax
  __int64 v45; // rdx
  unsigned __int64 v46; // rcx
  __int64 v47; // [rsp+0h] [rbp-C0h]
  _QWORD *v48; // [rsp+0h] [rbp-C0h]
  __int64 v49; // [rsp+0h] [rbp-C0h]
  __int64 v51; // [rsp+18h] [rbp-A8h]
  _DWORD *v52; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v53; // [rsp+28h] [rbp-98h]
  __int64 v54; // [rsp+30h] [rbp-90h]
  _BYTE *v55; // [rsp+40h] [rbp-80h] BYREF
  __int64 v56; // [rsp+48h] [rbp-78h]
  _BYTE v57[48]; // [rsp+50h] [rbp-70h] BYREF
  int v58; // [rsp+80h] [rbp-40h]

  v2 = a2;
  v3 = *(_QWORD **)(a2 + 32);
  v4 = (_DWORD *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  v51 = *(_QWORD *)a2;
  v5 = ((*(_WORD *)(*(_QWORD *)a2 + 2LL) >> 4) & 0x3FF) - 87;
  if ( v5 <= 9u && ((1LL << v5) & 0x35F) != 0 || !*(_QWORD *)(v51 + 16) )
    return 0;
  v54 = 0;
  v7 = v4[4];
  v52 = 0;
  v8 = *(_QWORD *)(a2 + 8);
  LODWORD(v55) = -1;
  v53 = 0;
  v9 = (unsigned int)(v7 + 31) >> 5;
  if ( v9 )
  {
    sub_1CFD340((__int64)&v52, 0, v9, &v55);
    v51 = *(_QWORD *)a2;
  }
  sub_2F7A480(*a1, v8);
  v10 = *(_QWORD *)(a2 + 16);
  v58 = 0;
  v55 = v57;
  v56 = 0x600000000LL;
  v11 = *(__int64 (**)())(*(_QWORD *)v10 + 136LL);
  if ( v11 == sub_2DD19D0 )
  {
    (*(void (**)(void))(*(_QWORD *)v10 + 200LL))();
    v58 = 0;
    LODWORD(v56) = 0;
    BUG();
  }
  v47 = v11();
  v12 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  v58 = 0;
  v13 = (_QWORD *)v12;
  LODWORD(v56) = 0;
  (*(void (__fastcall **)(__int64, __int64, _BYTE **))(*(_QWORD *)v47 + 256LL))(v47, a2, &v55);
  v16 = v55;
  v17 = 8LL * (unsigned int)v56;
  v18 = (unsigned __int64)&v55[v17];
  v19 = v17 >> 3;
  v20 = v17 >> 5;
  if ( v20 )
  {
    v20 = (__int64)&v55[32 * v20];
    while ( !*v16 )
    {
      if ( v16[1] )
      {
        ++v16;
        goto LABEL_15;
      }
      if ( v16[2] )
      {
        v16 += 2;
        goto LABEL_15;
      }
      if ( v16[3] )
      {
        v16 += 3;
        goto LABEL_15;
      }
      v16 += 4;
      if ( (_QWORD *)v20 == v16 )
      {
        v19 = (__int64)(v18 - (_QWORD)v16) >> 3;
        goto LABEL_25;
      }
    }
    goto LABEL_15;
  }
LABEL_25:
  if ( v19 == 2 )
    goto LABEL_67;
  if ( v19 == 3 )
  {
    if ( *v16 )
      goto LABEL_15;
    ++v16;
LABEL_67:
    if ( !*v16 )
    {
      ++v16;
      goto LABEL_69;
    }
LABEL_15:
    if ( (_QWORD *)v18 != v16 )
    {
      v19 = v2;
      v21 = (unsigned __int16 *)(*(__int64 (__fastcall **)(_QWORD *, __int64, __int64))(*v13 + 72LL))(v13, v2, v20);
      v18 = *v21;
      v15 = v21;
      if ( (_WORD)v18 )
      {
        v22 = 0;
        v14 = 1;
        do
        {
          v20 = (__int64)v55;
          v19 = (unsigned __int16)v18 >> 6;
          if ( (*(_QWORD *)&v55[8 * v19] & (1LL << v18)) != 0 )
          {
            v23 = (__int16 *)(v13[7] + 2LL * *(unsigned int *)(v13[1] + 24 * v18 + 4));
            v19 = (__int64)(v23 + 1);
            v24 = *v23 + (unsigned __int16)v18;
            if ( *v23 )
            {
              for ( i = v24; ; i = v24 )
              {
                v19 += 2;
                *(_QWORD *)(v20 + ((i >> 3) & 0x1FF8)) |= 1LL << i;
                if ( !*(_WORD *)(v19 - 2) )
                  break;
                v24 += *(__int16 *)(v19 - 2);
                v20 = (__int64)v55;
              }
            }
          }
          v18 = v15[++v22];
        }
        while ( (_WORD)v18 );
      }
    }
    goto LABEL_28;
  }
  if ( v19 != 1 )
    goto LABEL_28;
LABEL_69:
  if ( *v16 )
    goto LABEL_15;
LABEL_28:
  *v52 &= ~1u;
  v26 = *(__int64 (**)())(*(_QWORD *)v4 + 112LL);
  if ( v26 != sub_2F7B4B0 )
  {
    v19 = v2;
    v41 = ((__int64 (__fastcall *)(_DWORD *, __int64, __int64, unsigned __int64, __int64, unsigned __int16 *))v26)(
            v4,
            v2,
            v20,
            v18,
            v14,
            v15);
    if ( v41 != v41 + 2 * v20 )
    {
      v49 = v2;
      v42 = (unsigned __int16 *)v41;
      v43 = (unsigned __int16 *)(v41 + 2 * v20);
      do
      {
        v44 = sub_E922F0(v4, *v42);
        v19 = (__int64)&v44[2 * v45];
        v20 = (__int64)v44;
        if ( v44 != (char *)v19 )
        {
          do
          {
            v46 = *(unsigned __int16 *)v20;
            v20 += 2;
            *(_DWORD *)((char *)v52 + ((v46 >> 3) & 0x1FFC)) &= ~(1 << v46);
          }
          while ( v19 != v20 );
        }
        ++v42;
      }
      while ( v43 != v42 );
      v2 = v49;
    }
  }
  if ( v4[4] > 1u )
  {
    v27 = v4;
    v28 = 8;
    v29 = v4[4];
    v30 = 1;
    do
    {
      v31 = v30 >> 6;
      v20 = (__int64)v55;
      v19 = *(_QWORD *)&v55[8 * v31] & (1LL << v30);
      if ( !v19 )
      {
        if ( (v30 & 0x80000000) != 0 )
          v32 = *(_QWORD *)(v3[7] + 16LL * (v30 & 0x7FFFFFFF) + 8);
        else
          v32 = *(_QWORD *)(v3[38] + v28);
        if ( v32
          && ((*(_BYTE *)(v32 + 3) & 0x10) != 0
           || (v40 = *(_QWORD *)(v32 + 32)) != 0 && (*(_BYTE *)(v40 + 3) & 0x10) != 0) )
        {
          v19 = v30;
          v48 = v27;
          v33 = sub_E922F0(v27, v30);
          v27 = v48;
          v34 = &v33[2 * v20];
          if ( v33 != v34 )
          {
            do
            {
              while ( 1 )
              {
                v35 = *(unsigned __int16 *)v33;
                v36 = *(_WORD *)v33;
                v19 = v35 >> 6;
                v20 = *(_QWORD *)&v55[8 * v19] & (1LL << *(_WORD *)v33);
                if ( !v20 )
                  break;
                v33 += 2;
                if ( v34 == v33 )
                  goto LABEL_41;
              }
              v20 = (__int64)v52;
              v33 += 2;
              v19 = (unsigned int)(1 << v36);
              v52[v35 >> 5] &= ~(1 << v36);
            }
            while ( v34 != v33 );
LABEL_41:
            v27 = v48;
          }
        }
        else
        {
          v20 = v3[39];
          if ( (*(_QWORD *)(v20 + 8 * v31) & (1LL << v30)) != 0 )
          {
            v19 = (__int64)v52;
            v20 = (unsigned int)~(1 << v30);
            v52[v30 >> 5] &= v20;
          }
        }
      }
      ++v30;
      v28 += 8;
    }
    while ( v30 != v29 );
  }
  if ( (unsigned __int8)sub_2FDC130(v51, v19, v20) )
  {
    v37 = *(__int64 (**)())(**(_QWORD **)(v2 + 16) + 136LL);
    if ( v37 == sub_2DD19D0 )
      BUG();
    v38 = v37();
    v39 = *(__int64 (**)())(*(_QWORD *)v38 + 344LL);
    if ( v39 != sub_2F7B4C0 )
      ((void (__fastcall *)(__int64, __int64))v39)(v38, v51);
  }
  sub_2F7B1D0(*a1, v51, v52, (v53 - (__int64)v52) >> 2);
  if ( v55 != v57 )
    _libc_free((unsigned __int64)v55);
  if ( v52 )
    j_j___libc_free_0((unsigned __int64)v52);
  return 0;
}
