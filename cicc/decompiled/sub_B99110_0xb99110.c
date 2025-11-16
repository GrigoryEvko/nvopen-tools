// Function: sub_B99110
// Address: 0xb99110
//
void __fastcall sub_B99110(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r14d
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rsi
  unsigned int v8; // eax
  __int64 *v9; // r12
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // r13
  int v13; // eax
  __int64 v14; // rsi
  int v15; // edx
  unsigned int v16; // eax
  __int64 v17; // r14
  __int64 v18; // rcx
  __int64 v19; // r15
  __int64 v20; // r12
  __int64 v22; // r13
  unsigned int v23; // esi
  __int64 v24; // rcx
  int v25; // r10d
  __int64 *v26; // rdi
  unsigned int v27; // eax
  __int64 v28; // rdx
  __int64 v29; // r9
  __int64 *v30; // rdi
  int v31; // eax
  __int64 v32; // rdx
  _QWORD *v33; // rax
  int v34; // r8d
  int v35; // edi
  __int64 v36; // [rsp+0h] [rbp-40h] BYREF
  __int64 *v37; // [rsp+8h] [rbp-38h] BYREF

  v3 = a2;
  if ( a3 )
  {
    v22 = *(_QWORD *)sub_BD5C60(a1, a2);
    v36 = a1;
    v23 = *(_DWORD *)(v22 + 3248);
    if ( v23 )
    {
      v24 = *(_QWORD *)(v22 + 3232);
      v25 = 1;
      v26 = 0;
      v27 = (v23 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v28 = v24 + 40LL * v27;
      v29 = *(_QWORD *)v28;
      if ( a1 == *(_QWORD *)v28 )
      {
LABEL_20:
        v30 = (__int64 *)(v28 + 8);
        if ( *(_DWORD *)(v28 + 16) )
        {
LABEL_21:
          sub_B980E0(v30, v3, a3);
          return;
        }
LABEL_35:
        *(_BYTE *)(a1 + 7) |= 0x20u;
        goto LABEL_21;
      }
      while ( v29 != -4096 )
      {
        if ( !v26 && v29 == -8192 )
          v26 = (__int64 *)v28;
        v27 = (v23 - 1) & (v25 + v27);
        v28 = v24 + 40LL * v27;
        v29 = *(_QWORD *)v28;
        if ( a1 == *(_QWORD *)v28 )
          goto LABEL_20;
        ++v25;
      }
      v31 = *(_DWORD *)(v22 + 3240);
      if ( !v26 )
        v26 = (__int64 *)v28;
      ++*(_QWORD *)(v22 + 3224);
      v37 = v26;
      if ( 4 * (v31 + 1) < 3 * v23 )
      {
        v32 = a1;
        if ( v23 - *(_DWORD *)(v22 + 3244) - (v31 + 1) > v23 >> 3 )
        {
LABEL_32:
          ++*(_DWORD *)(v22 + 3240);
          if ( *v26 != -4096 )
            --*(_DWORD *)(v22 + 3244);
          v33 = v26 + 3;
          *v26 = v32;
          v30 = v26 + 1;
          *v30 = (__int64)v33;
          v30[1] = 0x100000000LL;
          goto LABEL_35;
        }
LABEL_39:
        sub_B98D30(v22 + 3224, v23);
        sub_B92880(v22 + 3224, &v36, &v37);
        v32 = v36;
        v26 = v37;
        goto LABEL_32;
      }
    }
    else
    {
      ++*(_QWORD *)(v22 + 3224);
      v37 = 0;
    }
    v23 *= 2;
    goto LABEL_39;
  }
  if ( (*(_BYTE *)(a1 + 7) & 0x20) == 0 )
    return;
  v5 = *(_QWORD *)sub_BD5C60(a1, a2);
  v6 = *(unsigned int *)(v5 + 3248);
  v7 = *(_QWORD *)(v5 + 3232);
  if ( (_DWORD)v6 )
  {
    v8 = (v6 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v9 = (__int64 *)(v7 + 40LL * v8);
    v10 = *v9;
    if ( a1 == *v9 )
      goto LABEL_6;
    v34 = 1;
    while ( v10 != -4096 )
    {
      v8 = (v6 - 1) & (v34 + v8);
      v9 = (__int64 *)(v7 + 40LL * v8);
      v10 = *v9;
      if ( a1 == *v9 )
        goto LABEL_6;
      ++v34;
    }
  }
  v9 = (__int64 *)(v7 + 40 * v6);
LABEL_6:
  sub_B97D20(v9 + 1, v3);
  if ( !*((_DWORD *)v9 + 4) )
  {
    v11 = sub_BD5C60(a1, v3);
    v12 = *(_QWORD *)v11;
    v13 = *(_DWORD *)(*(_QWORD *)v11 + 3248LL);
    v14 = *(_QWORD *)(v12 + 3232);
    if ( v13 )
    {
      v15 = v13 - 1;
      v16 = (v13 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v17 = v14 + 40LL * v16;
      v18 = *(_QWORD *)v17;
      if ( a1 == *(_QWORD *)v17 )
      {
LABEL_9:
        v19 = *(_QWORD *)(v17 + 8);
        v20 = v19 + 16LL * *(unsigned int *)(v17 + 16);
        if ( v19 != v20 )
        {
          do
          {
            v14 = *(_QWORD *)(v20 - 8);
            v20 -= 16;
            if ( v14 )
              sub_B91220(v20 + 8, v14);
          }
          while ( v19 != v20 );
          v20 = *(_QWORD *)(v17 + 8);
        }
        if ( v20 != v17 + 24 )
          _libc_free(v20, v14);
        *(_QWORD *)v17 = -8192;
        --*(_DWORD *)(v12 + 3240);
        ++*(_DWORD *)(v12 + 3244);
      }
      else
      {
        v35 = 1;
        while ( v18 != -4096 )
        {
          v16 = v15 & (v35 + v16);
          v17 = v14 + 40LL * v16;
          v18 = *(_QWORD *)v17;
          if ( a1 == *(_QWORD *)v17 )
            goto LABEL_9;
          ++v35;
        }
      }
    }
    *(_BYTE *)(a1 + 7) &= ~0x20u;
  }
}
