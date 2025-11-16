// Function: sub_3741D90
// Address: 0x3741d90
//
__int64 __fastcall sub_3741D90(__int64 a1, __int64 a2)
{
  __int64 (*v3)(); // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 *v7; // rax
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // rdx
  __int64 i; // r11
  __int64 v12; // r8
  unsigned int v13; // edi
  _QWORD *v14; // rax
  __int64 v15; // rcx
  _DWORD *v16; // rax
  unsigned int v17; // edx
  __int64 v18; // rcx
  unsigned int v19; // eax
  __int64 *v20; // r12
  __int64 v21; // rsi
  __int64 v22; // r13
  unsigned int v23; // esi
  int v24; // eax
  int v25; // esi
  __int64 v26; // rdi
  unsigned int v27; // eax
  int v28; // ecx
  _QWORD *v29; // rdx
  __int64 v30; // r8
  int v31; // r8d
  int v32; // eax
  int v33; // eax
  int v34; // eax
  _QWORD *v35; // r8
  int v36; // r9d
  unsigned int v37; // r15d
  __int64 v38; // rdi
  __int64 v39; // rsi
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 *v42; // rax
  int v43; // r10d
  _QWORD *v44; // r9
  __int64 v45; // [rsp+10h] [rbp-40h]
  int v46; // [rsp+10h] [rbp-40h]
  __int64 v47; // [rsp+10h] [rbp-40h]
  unsigned __int8 v48; // [rsp+1Fh] [rbp-31h]

  if ( !*(_BYTE *)(*(_QWORD *)(a1 + 40) + 48LL) )
    return 0;
  v3 = *(__int64 (**)())(*(_QWORD *)a1 + 32LL);
  if ( v3 == sub_3740EB0 )
    return 0;
  v48 = v3();
  if ( !v48 )
    return 0;
  v7 = *(__int64 **)(a1 + 40);
  v8 = *v7;
  if ( (*(_BYTE *)(*v7 + 2) & 1) != 0 )
  {
    sub_B2C6D0(*v7, a2, v5, v6);
    v42 = *(__int64 **)(a1 + 40);
    v9 = *(_QWORD *)(v8 + 96);
    v8 = *v42;
    if ( (*(_BYTE *)(*v42 + 2) & 1) != 0 )
      sub_B2C6D0(*v42, a2, v40, v41);
    v10 = *(_QWORD *)(v8 + 96);
  }
  else
  {
    v9 = *(_QWORD *)(v8 + 96);
    v10 = v9;
  }
  for ( i = v10 + 40LL * *(_QWORD *)(v8 + 104); v9 != i; *v16 = *((_DWORD *)v20 + 2) )
  {
    v17 = *(_DWORD *)(a1 + 32);
    v18 = *(_QWORD *)(a1 + 16);
    if ( v17 )
    {
      v19 = (v17 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v20 = (__int64 *)(v18 + 16LL * v19);
      v21 = *v20;
      if ( v9 == *v20 )
        goto LABEL_15;
      v31 = 1;
      while ( v21 != -4096 )
      {
        v19 = (v17 - 1) & (v31 + v19);
        v20 = (__int64 *)(v18 + 16LL * v19);
        v21 = *v20;
        if ( v9 == *v20 )
          goto LABEL_15;
        ++v31;
      }
    }
    v20 = (__int64 *)(v18 + 16LL * v17);
LABEL_15:
    v22 = *(_QWORD *)(a1 + 40);
    v23 = *(_DWORD *)(v22 + 144);
    if ( !v23 )
    {
      ++*(_QWORD *)(v22 + 120);
      goto LABEL_17;
    }
    v12 = *(_QWORD *)(v22 + 128);
    v13 = (v23 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    v14 = (_QWORD *)(v12 + 16LL * v13);
    v15 = *v14;
    if ( v9 != *v14 )
    {
      v46 = 1;
      v29 = 0;
      while ( v15 != -4096 )
      {
        if ( !v29 && v15 == -8192 )
          v29 = v14;
        v13 = (v23 - 1) & (v46 + v13);
        v14 = (_QWORD *)(v12 + 16LL * v13);
        v15 = *v14;
        if ( v9 == *v14 )
          goto LABEL_11;
        ++v46;
      }
      if ( !v29 )
        v29 = v14;
      v32 = *(_DWORD *)(v22 + 136);
      ++*(_QWORD *)(v22 + 120);
      v28 = v32 + 1;
      if ( 4 * (v32 + 1) >= 3 * v23 )
      {
LABEL_17:
        v45 = i;
        sub_3384500(v22 + 120, 2 * v23);
        v24 = *(_DWORD *)(v22 + 144);
        if ( !v24 )
          goto LABEL_62;
        v25 = v24 - 1;
        v26 = *(_QWORD *)(v22 + 128);
        i = v45;
        v27 = (v24 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v28 = *(_DWORD *)(v22 + 136) + 1;
        v29 = (_QWORD *)(v26 + 16LL * v27);
        v30 = *v29;
        if ( v9 != *v29 )
        {
          v43 = 1;
          v44 = 0;
          while ( v30 != -4096 )
          {
            if ( v30 == -8192 && !v44 )
              v44 = v29;
            v27 = v25 & (v43 + v27);
            v29 = (_QWORD *)(v26 + 16LL * v27);
            v30 = *v29;
            if ( v9 == *v29 )
              goto LABEL_19;
            ++v43;
          }
          if ( v44 )
            v29 = v44;
        }
      }
      else if ( v23 - *(_DWORD *)(v22 + 140) - v28 <= v23 >> 3 )
      {
        v47 = i;
        sub_3384500(v22 + 120, v23);
        v33 = *(_DWORD *)(v22 + 144);
        if ( !v33 )
        {
LABEL_62:
          ++*(_DWORD *)(v22 + 136);
          BUG();
        }
        v34 = v33 - 1;
        v35 = 0;
        i = v47;
        v36 = 1;
        v37 = v34 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v38 = *(_QWORD *)(v22 + 128);
        v28 = *(_DWORD *)(v22 + 136) + 1;
        v29 = (_QWORD *)(v38 + 16LL * v37);
        v39 = *v29;
        if ( v9 != *v29 )
        {
          while ( v39 != -4096 )
          {
            if ( !v35 && v39 == -8192 )
              v35 = v29;
            v37 = v34 & (v36 + v37);
            v29 = (_QWORD *)(v38 + 16LL * v37);
            v39 = *v29;
            if ( v9 == *v29 )
              goto LABEL_19;
            ++v36;
          }
          if ( v35 )
            v29 = v35;
        }
      }
LABEL_19:
      *(_DWORD *)(v22 + 136) = v28;
      if ( *v29 != -4096 )
        --*(_DWORD *)(v22 + 140);
      *v29 = v9;
      v16 = v29 + 1;
      *((_DWORD *)v29 + 2) = 0;
      goto LABEL_12;
    }
LABEL_11:
    v16 = v14 + 1;
LABEL_12:
    v9 += 40;
  }
  return v48;
}
