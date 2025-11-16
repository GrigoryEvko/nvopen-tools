// Function: sub_2740DD0
// Address: 0x2740dd0
//
__int64 __fastcall sub_2740DD0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rdi
  unsigned int v11; // ecx
  __int64 v12; // rdx
  int v14; // edx
  __int64 v15; // rax
  __int64 v16; // r14
  char v17; // dl
  __int64 v18; // rcx
  int v19; // esi
  unsigned int v20; // eax
  __int64 v21; // r13
  __int64 v22; // rdi
  unsigned int v23; // esi
  int v24; // r10d
  unsigned int v25; // eax
  int v26; // ecx
  unsigned int v27; // edx
  __int64 v28; // r12
  __int64 v29; // rax
  __int64 v30; // [rsp+8h] [rbp-38h] BYREF
  __int64 v31; // [rsp+10h] [rbp-30h] BYREF
  int v32; // [rsp+18h] [rbp-28h]

  v8 = *a1;
  v9 = *(unsigned int *)(*a1 + 24LL);
  v10 = *(_QWORD *)(*a1 + 8LL);
  if ( (_DWORD)v9 )
  {
    a5 = (unsigned int)(v9 - 1);
    v11 = a5 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v12 = v10 + 16LL * v11;
    a6 = *(_QWORD *)v12;
    if ( a2 == *(_QWORD *)v12 )
    {
LABEL_3:
      if ( v12 != v10 + 16 * v9 )
        return *(unsigned int *)(v12 + 8);
    }
    else
    {
      v14 = 1;
      while ( a6 != -4096 )
      {
        v24 = v14 + 1;
        v11 = a5 & (v14 + v11);
        v12 = v10 + 16LL * v11;
        a6 = *(_QWORD *)v12;
        if ( a2 == *(_QWORD *)v12 )
          goto LABEL_3;
        v14 = v24;
      }
    }
  }
  v15 = a1[1];
  v16 = a1[2];
  v31 = a2;
  v32 = *(_DWORD *)(v15 + 8) + *(_DWORD *)(v8 + 16) + 1;
  v17 = *(_BYTE *)(v16 + 8) & 1;
  if ( v17 )
  {
    v18 = v16 + 16;
    v19 = 3;
  }
  else
  {
    v23 = *(_DWORD *)(v16 + 24);
    v18 = *(_QWORD *)(v16 + 16);
    if ( !v23 )
    {
      v25 = *(_DWORD *)(v16 + 8);
      v30 = 0;
      ++*(_QWORD *)v16;
      v26 = (v25 >> 1) + 1;
      goto LABEL_16;
    }
    v19 = v23 - 1;
  }
  v20 = v19 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v21 = v18 + 16LL * v20;
  v22 = *(_QWORD *)v21;
  if ( a2 != *(_QWORD *)v21 )
  {
    a6 = 1;
    a5 = 0;
    while ( v22 != -4096 )
    {
      if ( v22 == -8192 && !a5 )
        a5 = v21;
      v20 = v19 & (a6 + v20);
      v21 = v18 + 16LL * v20;
      v22 = *(_QWORD *)v21;
      if ( a2 == *(_QWORD *)v21 )
        return *(unsigned int *)(v21 + 8);
      a6 = (unsigned int)(a6 + 1);
    }
    v25 = *(_DWORD *)(v16 + 8);
    if ( !a5 )
      a5 = v21;
    v30 = a5;
    v26 = (v25 >> 1) + 1;
    ++*(_QWORD *)v16;
    if ( v17 )
    {
      v27 = 12;
      v23 = 4;
LABEL_17:
      if ( 4 * v26 >= v27 )
      {
        v23 *= 2;
      }
      else if ( v23 - *(_DWORD *)(v16 + 12) - v26 > v23 >> 3 )
      {
LABEL_19:
        v21 = v30;
        *(_DWORD *)(v16 + 8) = (2 * (v25 >> 1) + 2) | v25 & 1;
        if ( *(_QWORD *)v21 != -4096 )
          --*(_DWORD *)(v16 + 12);
        *(_QWORD *)v21 = v31;
        *(_DWORD *)(v21 + 8) = v32;
        v28 = a1[1];
        v29 = *(unsigned int *)(v28 + 8);
        if ( v29 + 1 > (unsigned __int64)*(unsigned int *)(v28 + 12) )
        {
          sub_C8D5F0(v28, (const void *)(v28 + 16), v29 + 1, 8u, a5, a6);
          v29 = *(unsigned int *)(v28 + 8);
        }
        *(_QWORD *)(*(_QWORD *)v28 + 8 * v29) = a2;
        ++*(_DWORD *)(v28 + 8);
        return *(unsigned int *)(v21 + 8);
      }
      sub_BB64D0(v16, v23);
      sub_27400A0(v16, &v31, &v30);
      v25 = *(_DWORD *)(v16 + 8);
      goto LABEL_19;
    }
    v23 = *(_DWORD *)(v16 + 24);
LABEL_16:
    v27 = 3 * v23;
    goto LABEL_17;
  }
  return *(unsigned int *)(v21 + 8);
}
