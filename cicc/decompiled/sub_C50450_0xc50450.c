// Function: sub_C50450
// Address: 0xc50450
//
void __fastcall sub_C50450(__int64 **a1, __int64 a2, char a3)
{
  unsigned int v5; // edx
  __int64 v6; // rax
  __int64 *v7; // r15
  __int64 v8; // r14
  __int64 v9; // rsi
  int v10; // eax
  char *v11; // rax
  char *v12; // rdx
  __int64 v13; // rdx
  __int64 *v14; // rax
  char v15; // dl
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rax
  __int64 *v19; // rax
  signed __int64 v20; // rsi
  __int64 v21; // [rsp+0h] [rbp-160h]
  __int64 v22; // [rsp+8h] [rbp-158h]
  __int64 v23; // [rsp+10h] [rbp-150h] BYREF
  char *v24; // [rsp+18h] [rbp-148h]
  __int64 v25; // [rsp+20h] [rbp-140h]
  int v26; // [rsp+28h] [rbp-138h]
  char v27; // [rsp+2Ch] [rbp-134h]
  char v28; // [rsp+30h] [rbp-130h] BYREF

  v5 = *((_DWORD *)a1 + 2);
  v23 = 0;
  v24 = &v28;
  v25 = 32;
  v26 = 0;
  v27 = 1;
  if ( v5 )
  {
    v6 = **a1;
    v7 = *a1;
    if ( v6 != -8 )
      goto LABEL_4;
    do
    {
      do
      {
        v6 = v7[1];
        ++v7;
      }
      while ( v6 == -8 );
LABEL_4:
      ;
    }
    while ( !v6 );
    v8 = (__int64)&(*a1)[v5];
    if ( (__int64 *)v8 != v7 )
    {
      while ( 1 )
      {
        v9 = *(_QWORD *)(*v7 + 8);
        v10 = (*(_BYTE *)(v9 + 12) >> 5) & 3;
        if ( v10 == 2 || v10 == 1 && !a3 )
          goto LABEL_14;
        if ( !v27 )
          goto LABEL_19;
        v11 = v24;
        v12 = &v24[8 * HIDWORD(v25)];
        if ( v24 == v12 )
          break;
        while ( v9 != *(_QWORD *)v11 )
        {
          v11 += 8;
          if ( v12 == v11 )
            goto LABEL_23;
        }
LABEL_14:
        v13 = v7[1];
        v14 = v7 + 1;
        if ( v13 != -8 )
          goto LABEL_16;
        do
        {
          do
          {
            v13 = v14[1];
            ++v14;
          }
          while ( v13 == -8 );
LABEL_16:
          ;
        }
        while ( !v13 );
        if ( v14 == (__int64 *)v8 )
          goto LABEL_25;
        v7 = v14;
      }
LABEL_23:
      if ( HIDWORD(v25) >= (unsigned int)v25 )
      {
LABEL_19:
        sub_C8CC70(&v23, v9);
        if ( !v15 )
          goto LABEL_14;
      }
      else
      {
        ++HIDWORD(v25);
        *(_QWORD *)v12 = v9;
        ++v23;
      }
      v16 = *(_QWORD *)(*v7 + 8);
      v17 = *v7 + 16;
      v18 = *(unsigned int *)(a2 + 8);
      if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        v21 = *v7 + 16;
        v22 = *(_QWORD *)(*v7 + 8);
        sub_C8D5F0(a2, a2 + 16, v18 + 1, 16);
        v18 = *(unsigned int *)(a2 + 8);
        v17 = v21;
        v16 = v22;
      }
      v19 = (__int64 *)(*(_QWORD *)a2 + 16 * v18);
      *v19 = v17;
      v19[1] = v16;
      ++*(_DWORD *)(a2 + 8);
      goto LABEL_14;
    }
  }
LABEL_25:
  v20 = 16LL * *(unsigned int *)(a2 + 8);
  if ( *(unsigned int *)(a2 + 8) > 1uLL )
  {
    v20 >>= 4;
    qsort(*(void **)a2, v20, 0x10u, (__compar_fn_t)sub_C4F850);
  }
  if ( !v27 )
    _libc_free(v24, v20);
}
