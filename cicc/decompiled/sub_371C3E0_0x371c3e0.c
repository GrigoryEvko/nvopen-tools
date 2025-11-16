// Function: sub_371C3E0
// Address: 0x371c3e0
//
unsigned __int64 __fastcall sub_371C3E0(__int64 a1, __int64 a2)
{
  unsigned int v4; // esi
  __int64 v5; // r8
  __int64 v6; // rdi
  int v7; // r10d
  _QWORD *v8; // r9
  unsigned int v9; // ecx
  _QWORD *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rax
  int v17; // edx
  unsigned __int64 result; // rax
  int v19; // eax
  int v20; // edx
  __int64 v21; // rax
  int v22; // eax
  int v23; // ecx
  __int64 v24; // rdi
  unsigned int v25; // eax
  __int64 v26; // rsi
  int v27; // r10d
  int v28; // eax
  int v29; // eax
  __int64 v30; // rsi
  unsigned int v31; // r13d
  _QWORD *v32; // rdi
  __int64 v33; // rcx
  bool v34; // cc

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_23;
  }
  v5 = v4 - 1;
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 1;
  v8 = 0;
  v9 = v5 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (_QWORD *)(v6 + 8LL * v9);
  v11 = *v10;
  if ( *v10 == a2 )
    goto LABEL_3;
  while ( v11 != -4096 )
  {
    if ( v8 || v11 != -8192 )
      v10 = v8;
    v9 = v5 & (v7 + v9);
    v11 = *(_QWORD *)(v6 + 8LL * v9);
    if ( v11 == a2 )
      goto LABEL_3;
    ++v7;
    v8 = v10;
    v10 = (_QWORD *)(v6 + 8LL * v9);
  }
  v19 = *(_DWORD *)(a1 + 16);
  if ( !v8 )
    v8 = v10;
  ++*(_QWORD *)a1;
  v20 = v19 + 1;
  if ( 4 * (v19 + 1) >= 3 * v4 )
  {
LABEL_23:
    sub_31B3C80(a1, 2 * v4);
    v22 = *(_DWORD *)(a1 + 24);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a1 + 8);
      v25 = (v22 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v8 = (_QWORD *)(v24 + 8LL * v25);
      v26 = *v8;
      v20 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v8 != a2 )
      {
        v27 = 1;
        v5 = 0;
        while ( v26 != -4096 )
        {
          if ( !v5 && v26 == -8192 )
            v5 = (__int64)v8;
          v25 = v23 & (v27 + v25);
          v8 = (_QWORD *)(v24 + 8LL * v25);
          v26 = *v8;
          if ( *v8 == a2 )
            goto LABEL_17;
          ++v27;
        }
        if ( v5 )
          v8 = (_QWORD *)v5;
      }
      goto LABEL_17;
    }
    goto LABEL_50;
  }
  if ( v4 - *(_DWORD *)(a1 + 20) - v20 <= v4 >> 3 )
  {
    sub_31B3C80(a1, v4);
    v28 = *(_DWORD *)(a1 + 24);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = *(_QWORD *)(a1 + 8);
      v5 = 1;
      v31 = v29 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v8 = (_QWORD *)(v30 + 8LL * v31);
      v20 = *(_DWORD *)(a1 + 16) + 1;
      v32 = 0;
      v33 = *v8;
      if ( *v8 != a2 )
      {
        while ( v33 != -4096 )
        {
          if ( v33 == -8192 && !v32 )
            v32 = v8;
          v31 = v29 & (v5 + v31);
          v8 = (_QWORD *)(v30 + 8LL * v31);
          v33 = *v8;
          if ( *v8 == a2 )
            goto LABEL_17;
          v5 = (unsigned int)(v5 + 1);
        }
        if ( v32 )
          v8 = v32;
      }
      goto LABEL_17;
    }
LABEL_50:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_17:
  *(_DWORD *)(a1 + 16) = v20;
  if ( *v8 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v8 = a2;
  v21 = *(unsigned int *)(a1 + 40);
  if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v21 + 1, 8u, v5, (__int64)v8);
    v21 = *(unsigned int *)(a1 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v21) = a2;
  ++*(_DWORD *)(a1 + 40);
LABEL_3:
  sub_B9A090(*(_QWORD *)(a2 + 16), "sandboxvec", 0xAu, *(_QWORD *)(a1 + 112));
  v16 = sub_371B7D0(a1 + 128, a2, v12, v13, v14, v15);
  if ( v17 == 1 )
    *(_DWORD *)(a1 + 152) = 1;
  if ( __OFADD__(*(_QWORD *)(a1 + 144), v16) )
  {
    v34 = v16 <= 0;
    result = 0x7FFFFFFFFFFFFFFFLL;
    if ( v34 )
      result = 0x8000000000000000LL;
  }
  else
  {
    result = *(_QWORD *)(a1 + 144) + v16;
  }
  *(_QWORD *)(a1 + 144) = result;
  return result;
}
