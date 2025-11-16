// Function: sub_9BEC40
// Address: 0x9bec40
//
__int64 __fastcall sub_9BEC40(__int64 a1, __int64 a2, int a3, char a4)
{
  __int64 v4; // r9
  unsigned int v9; // esi
  __int64 v10; // rdi
  int v11; // r11d
  _QWORD *v12; // r10
  unsigned int v13; // ecx
  _QWORD *v14; // r14
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r8
  int v18; // edx
  _DWORD *v19; // rax
  int v20; // esi
  __int64 *v21; // rax
  __int64 v22; // rcx
  __int64 *v23; // rdx
  __int64 v24; // r9
  int v26; // ecx
  int v27; // ecx
  int v28; // edx
  _DWORD *v29; // rcx
  int v30; // r11d
  _DWORD *v31; // r9
  unsigned int v32; // edi
  int v33; // eax
  int v34; // edi
  __int64 v35; // rsi
  unsigned int v36; // eax
  __int64 v37; // rdx
  int v38; // r9d
  _QWORD *v39; // r8
  int v40; // edx
  int v41; // r9d
  __int64 v42; // r8
  _QWORD *v43; // rdx
  __int64 v44; // rax
  int v45; // esi
  __int64 v46; // rdi
  __int64 v47; // [rsp+8h] [rbp-38h]
  unsigned int v48; // [rsp+8h] [rbp-38h]

  v4 = a1 + 48;
  v9 = *(_DWORD *)(a1 + 72);
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 48);
    goto LABEL_37;
  }
  v10 = *(_QWORD *)(a1 + 56);
  v11 = 1;
  v12 = 0;
  v13 = (v9 - 1) & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
  v14 = (_QWORD *)(v10 + 16LL * v13);
  v15 = *v14;
  if ( *v14 == a2 )
    goto LABEL_3;
  while ( v15 != -4096 )
  {
    if ( !v12 && v15 == -8192 )
      v12 = v14;
    v13 = (v9 - 1) & (v11 + v13);
    v14 = (_QWORD *)(v10 + 16LL * v13);
    v15 = *v14;
    if ( *v14 == a2 )
      goto LABEL_3;
    ++v11;
  }
  v26 = *(_DWORD *)(a1 + 64);
  if ( v12 )
    v14 = v12;
  ++*(_QWORD *)(a1 + 48);
  v27 = v26 + 1;
  if ( 4 * v27 >= 3 * v9 )
  {
LABEL_37:
    sub_9BB560(v4, 2 * v9);
    v33 = *(_DWORD *)(a1 + 72);
    if ( v33 )
    {
      v34 = v33 - 1;
      v35 = *(_QWORD *)(a1 + 56);
      v36 = (v33 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v27 = *(_DWORD *)(a1 + 64) + 1;
      v14 = (_QWORD *)(v35 + 16LL * v36);
      v37 = *v14;
      if ( *v14 != a2 )
      {
        v38 = 1;
        v39 = 0;
        while ( v37 != -4096 )
        {
          if ( !v39 && v37 == -8192 )
            v39 = v14;
          v36 = v34 & (v38 + v36);
          v14 = (_QWORD *)(v35 + 16LL * v36);
          v37 = *v14;
          if ( *v14 == a2 )
            goto LABEL_28;
          ++v38;
        }
        if ( v39 )
          v14 = v39;
      }
      goto LABEL_28;
    }
    goto LABEL_67;
  }
  if ( v9 - *(_DWORD *)(a1 + 68) - v27 <= v9 >> 3 )
  {
    v48 = ((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9);
    sub_9BB560(v4, v9);
    v40 = *(_DWORD *)(a1 + 72);
    if ( v40 )
    {
      v41 = v40 - 1;
      v42 = *(_QWORD *)(a1 + 56);
      v43 = 0;
      LODWORD(v44) = v41 & v48;
      v27 = *(_DWORD *)(a1 + 64) + 1;
      v45 = 1;
      v14 = (_QWORD *)(v42 + 16LL * (v41 & v48));
      v46 = *v14;
      if ( *v14 != a2 )
      {
        while ( v46 != -4096 )
        {
          if ( v46 == -8192 && !v43 )
            v43 = v14;
          v44 = v41 & (unsigned int)(v44 + v45);
          v14 = (_QWORD *)(v42 + 16 * v44);
          v46 = *v14;
          if ( *v14 == a2 )
            goto LABEL_28;
          ++v45;
        }
        if ( v43 )
          v14 = v43;
      }
      goto LABEL_28;
    }
LABEL_67:
    ++*(_DWORD *)(a1 + 64);
    BUG();
  }
LABEL_28:
  *(_DWORD *)(a1 + 64) = v27;
  if ( *v14 != -4096 )
    --*(_DWORD *)(a1 + 68);
  *v14 = a2;
  v14[1] = 0;
LABEL_3:
  v16 = sub_22077B0(56);
  v17 = v16;
  if ( v16 )
  {
    *(_BYTE *)(v16 + 5) = a4;
    *(_QWORD *)(v16 + 40) = 0;
    *(_QWORD *)(v16 + 48) = a2;
    *(_QWORD *)(v16 + 8) = 1;
    *(_QWORD *)(v16 + 16) = 0;
    *(_QWORD *)(v16 + 24) = 0;
    *(_DWORD *)v16 = a3 ^ (a3 >> 31);
    *(_DWORD *)v16 -= a3 >> 31;
    *(_BYTE *)(v16 + 4) = a3 < 0;
    *(_DWORD *)(v16 + 32) = 0;
    v47 = v16;
    sub_9BE710(v16 + 8, 0);
    v17 = v47;
    v18 = *(_DWORD *)(v47 + 32);
    if ( !v18 )
    {
      ++*(_DWORD *)(v47 + 24);
      BUG();
    }
    v19 = *(_DWORD **)(v47 + 16);
    v20 = *v19;
    if ( *v19 )
    {
      v28 = v18 - 1;
      v29 = *(_DWORD **)(v47 + 16);
      v30 = 1;
      v31 = 0;
      v32 = 0;
      while ( v20 != 0x7FFFFFFF )
      {
        if ( v20 == 0x80000000 && !v31 )
          v31 = v29;
        v32 = v28 & (v30 + v32);
        v29 = &v19[4 * v32];
        v20 = *v29;
        if ( !*v29 )
        {
          v19 += 4 * v32;
          goto LABEL_6;
        }
        ++v30;
      }
      v19 = v29;
      if ( v31 )
        v19 = v31;
    }
LABEL_6:
    ++*(_DWORD *)(v47 + 24);
    if ( *v19 != 0x7FFFFFFF )
      --*(_DWORD *)(v47 + 28);
    *v19 = 0;
    *((_QWORD *)v19 + 1) = a2;
  }
  v14[1] = v17;
  if ( !*(_BYTE *)(a1 + 108) )
  {
LABEL_16:
    sub_C8CC70(a1 + 80, v17);
    return v14[1];
  }
  v21 = *(__int64 **)(a1 + 88);
  v22 = *(unsigned int *)(a1 + 100);
  v23 = &v21[v22];
  if ( v21 == v23 )
  {
LABEL_15:
    if ( (unsigned int)v22 < *(_DWORD *)(a1 + 96) )
    {
      *(_DWORD *)(a1 + 100) = v22 + 1;
      *v23 = v17;
      ++*(_QWORD *)(a1 + 80);
      return v14[1];
    }
    goto LABEL_16;
  }
  while ( 1 )
  {
    v24 = *v21;
    if ( v17 == *v21 )
      return v24;
    if ( v23 == ++v21 )
      goto LABEL_15;
  }
}
