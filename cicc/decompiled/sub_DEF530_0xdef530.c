// Function: sub_DEF530
// Address: 0xdef530
//
__int64 __fastcall sub_DEF530(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rdi
  __int64 v7; // rsi
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r12
  __int64 *v11; // r15
  __int64 *i; // r14
  __int64 v13; // rsi
  int v14; // r15d
  __int64 *v15; // rax
  __int64 *v16; // r13
  __int64 v17; // r9
  int v18; // r11d
  __int64 **v19; // rdx
  unsigned int v20; // ecx
  unsigned int v21; // r8d
  _QWORD *v22; // rax
  __int64 *v23; // rdi
  _QWORD *v24; // rax
  int v26; // eax
  int v27; // edi
  int v28; // eax
  int v29; // ecx
  __int64 v30; // r8
  unsigned int v31; // eax
  int v32; // r10d
  __int64 **v33; // r9
  int v34; // eax
  int v35; // eax
  __int64 v36; // r8
  int v37; // r10d
  unsigned int v38; // ecx
  unsigned int v39; // [rsp+4h] [rbp-6Ch]
  __int64 *v40; // [rsp+10h] [rbp-60h] BYREF
  __int64 v41; // [rsp+18h] [rbp-58h]
  _BYTE v42[80]; // [rsp+20h] [rbp-50h] BYREF

  v4 = sub_DEEF40(a1, a2);
  v5 = *(_QWORD *)(a1 + 120);
  v6 = *(_QWORD *)(a1 + 112);
  v7 = v4;
  v40 = (__int64 *)v42;
  v41 = 0x400000000LL;
  v10 = sub_DEAB30(v6, v4, v5, (__int64)&v40, v8, v9);
  if ( v10 )
  {
    v11 = &v40[(unsigned int)v41];
    for ( i = v40; v11 != i; ++i )
    {
      v13 = *i;
      sub_DEF380(a1, v13);
    }
    v14 = *(_DWORD *)(a1 + 136);
    v15 = sub_DD8400(*(_QWORD *)(a1 + 112), a2);
    v7 = *(unsigned int *)(a1 + 24);
    v16 = v15;
    if ( (_DWORD)v7 )
    {
      v17 = *(_QWORD *)(a1 + 8);
      v18 = 1;
      v19 = 0;
      v20 = ((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4);
      v21 = (v7 - 1) & v20;
      v22 = (_QWORD *)(v17 + 24LL * v21);
      v23 = (__int64 *)*v22;
      if ( v16 == (__int64 *)*v22 )
      {
LABEL_6:
        v24 = v22 + 1;
LABEL_7:
        *(_DWORD *)v24 = v14;
        v24[1] = v10;
        goto LABEL_8;
      }
      while ( v23 != (__int64 *)-4096LL )
      {
        if ( !v19 && v23 == (__int64 *)-8192LL )
          v19 = (__int64 **)v22;
        v21 = (v7 - 1) & (v18 + v21);
        v22 = (_QWORD *)(v17 + 24LL * v21);
        v23 = (__int64 *)*v22;
        if ( v16 == (__int64 *)*v22 )
          goto LABEL_6;
        ++v18;
      }
      if ( !v19 )
        v19 = (__int64 **)v22;
      v26 = *(_DWORD *)(a1 + 16);
      ++*(_QWORD *)a1;
      v27 = v26 + 1;
      if ( 4 * (v26 + 1) < (unsigned int)(3 * v7) )
      {
        if ( (int)v7 - *(_DWORD *)(a1 + 20) - v27 > (unsigned int)v7 >> 3 )
        {
LABEL_21:
          *(_DWORD *)(a1 + 16) = v27;
          if ( *v19 != (__int64 *)-4096LL )
            --*(_DWORD *)(a1 + 20);
          *v19 = v16;
          v24 = v19 + 1;
          *((_DWORD *)v19 + 2) = 0;
          v19[2] = 0;
          goto LABEL_7;
        }
        v39 = v20;
        sub_DB02E0(a1, v7);
        v34 = *(_DWORD *)(a1 + 24);
        if ( v34 )
        {
          v35 = v34 - 1;
          v36 = *(_QWORD *)(a1 + 8);
          v37 = 1;
          v33 = 0;
          v38 = v35 & v39;
          v27 = *(_DWORD *)(a1 + 16) + 1;
          v19 = (__int64 **)(v36 + 24LL * (v35 & v39));
          v7 = (__int64)*v19;
          if ( v16 == *v19 )
            goto LABEL_21;
          while ( v7 != -4096 )
          {
            if ( !v33 && v7 == -8192 )
              v33 = v19;
            v38 = v35 & (v37 + v38);
            v19 = (__int64 **)(v36 + 24LL * v38);
            v7 = (__int64)*v19;
            if ( v16 == *v19 )
              goto LABEL_21;
            ++v37;
          }
          goto LABEL_29;
        }
        goto LABEL_45;
      }
    }
    else
    {
      ++*(_QWORD *)a1;
    }
    sub_DB02E0(a1, 2 * v7);
    v28 = *(_DWORD *)(a1 + 24);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = *(_QWORD *)(a1 + 8);
      v31 = (v28 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v27 = *(_DWORD *)(a1 + 16) + 1;
      v19 = (__int64 **)(v30 + 24LL * v31);
      v7 = (__int64)*v19;
      if ( v16 == *v19 )
        goto LABEL_21;
      v32 = 1;
      v33 = 0;
      while ( v7 != -4096 )
      {
        if ( !v33 && v7 == -8192 )
          v33 = v19;
        v31 = v29 & (v32 + v31);
        v19 = (__int64 **)(v30 + 24LL * v31);
        v7 = (__int64)*v19;
        if ( v16 == *v19 )
          goto LABEL_21;
        ++v32;
      }
LABEL_29:
      if ( v33 )
        v19 = v33;
      goto LABEL_21;
    }
LABEL_45:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_8:
  if ( v40 != (__int64 *)v42 )
    _libc_free(v40, v7);
  return v10;
}
