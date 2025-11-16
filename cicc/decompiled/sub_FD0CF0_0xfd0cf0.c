// Function: sub_FD0CF0
// Address: 0xfd0cf0
//
__int64 __fastcall sub_FD0CF0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r8d
  __int64 v4; // r9
  int v5; // r11d
  unsigned int v7; // edi
  unsigned int v9; // r12d
  unsigned int v10; // esi
  __int64 v11; // r15
  __int64 *v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r10
  unsigned int v15; // eax
  char v16; // cl
  __int64 v17; // rax
  int v19; // r11d
  __int64 *v20; // r10
  __int64 v21; // rdi
  int v22; // eax
  int v23; // ecx
  int v24; // eax
  int v25; // eax
  __int64 v26; // rdi
  unsigned int v27; // r12d
  __int64 v28; // rsi
  int v29; // r9d
  __int64 *v30; // r8
  int v31; // eax
  int v32; // eax
  __int64 v33; // rdi
  unsigned int v34; // r12d
  int v35; // r9d
  __int64 v36; // rsi
  __int64 v37; // [rsp-40h] [rbp-40h]
  __int64 v38; // [rsp-40h] [rbp-40h]

  v3 = *(_DWORD *)(a1 + 80);
  v4 = *(_QWORD *)(a1 + 64);
  if ( !v3 )
    return 0;
  v5 = 1;
  v7 = v3 - 1;
  v9 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v10 = (v3 - 1) & v9;
  LODWORD(v11) = v10;
  v12 = (__int64 *)(v4 + 16LL * v10);
  v13 = *v12;
  v14 = *v12;
  if ( *v12 == a2 )
  {
LABEL_3:
    v15 = *((_DWORD *)v12 + 2);
    v16 = v15 & 0x3F;
    v17 = 8LL * (v15 >> 6);
  }
  else
  {
    while ( 1 )
    {
      if ( v14 == -4096 )
        return 0;
      v11 = v7 & ((_DWORD)v11 + v5);
      v14 = *(_QWORD *)(v4 + 16 * v11);
      if ( v14 == a2 )
        break;
      ++v5;
    }
    v19 = 1;
    v20 = 0;
    while ( v13 != -4096 )
    {
      if ( !v20 && v13 == -8192 )
        v20 = v12;
      v10 = v7 & (v19 + v10);
      v12 = (__int64 *)(v4 + 16LL * v10);
      v13 = *v12;
      if ( *v12 == a2 )
        goto LABEL_3;
      ++v19;
    }
    v21 = a1 + 56;
    if ( !v20 )
      v20 = v12;
    v22 = *(_DWORD *)(a1 + 72);
    ++*(_QWORD *)(a1 + 56);
    v23 = v22 + 1;
    if ( 4 * (v22 + 1) < 3 * v3 )
    {
      if ( v3 - *(_DWORD *)(a1 + 76) - v23 > v3 >> 3 )
        goto LABEL_17;
      v38 = a3;
      sub_CE2410(v21, v3);
      v31 = *(_DWORD *)(a1 + 80);
      if ( v31 )
      {
        v32 = v31 - 1;
        v33 = *(_QWORD *)(a1 + 64);
        v30 = 0;
        v34 = v32 & v9;
        a3 = v38;
        v35 = 1;
        v23 = *(_DWORD *)(a1 + 72) + 1;
        v20 = (__int64 *)(v33 + 16LL * v34);
        v36 = *v20;
        if ( *v20 != a2 )
        {
          while ( v36 != -4096 )
          {
            if ( !v30 && v36 == -8192 )
              v30 = v20;
            v34 = v32 & (v35 + v34);
            v20 = (__int64 *)(v33 + 16LL * v34);
            v36 = *v20;
            if ( *v20 == a2 )
              goto LABEL_17;
            ++v35;
          }
          goto LABEL_24;
        }
        goto LABEL_17;
      }
LABEL_45:
      ++*(_DWORD *)(a1 + 72);
      BUG();
    }
    v37 = a3;
    sub_CE2410(v21, 2 * v3);
    v24 = *(_DWORD *)(a1 + 80);
    if ( !v24 )
      goto LABEL_45;
    v25 = v24 - 1;
    v26 = *(_QWORD *)(a1 + 64);
    v27 = v25 & v9;
    a3 = v37;
    v23 = *(_DWORD *)(a1 + 72) + 1;
    v20 = (__int64 *)(v26 + 16LL * v27);
    v28 = *v20;
    if ( *v20 != a2 )
    {
      v29 = 1;
      v30 = 0;
      while ( v28 != -4096 )
      {
        if ( v28 == -8192 && !v30 )
          v30 = v20;
        v27 = v25 & (v29 + v27);
        v20 = (__int64 *)(v26 + 16LL * v27);
        v28 = *v20;
        if ( *v20 == a2 )
          goto LABEL_17;
        ++v29;
      }
LABEL_24:
      if ( v30 )
        v20 = v30;
    }
LABEL_17:
    *(_DWORD *)(a1 + 72) = v23;
    if ( *v20 != -4096 )
      --*(_DWORD *)(a1 + 76);
    *v20 = a2;
    v16 = 0;
    v17 = 0;
    *((_DWORD *)v20 + 2) = 0;
  }
  return (*(_QWORD *)(*(_QWORD *)(a3 + 96) + v17) >> v16) & 1LL;
}
