// Function: sub_261D670
// Address: 0x261d670
//
__int64 __fastcall sub_261D670(__int64 a1, __int64 a2)
{
  int v3; // r14d
  __int64 v4; // r15
  int v5; // r14d
  int v6; // eax
  size_t v7; // rdx
  const void *v8; // rdi
  int v9; // ecx
  unsigned int i; // r13d
  __int64 v11; // r12
  const void *v12; // rsi
  int v13; // eax
  __int64 v14; // rdx
  unsigned int v16; // r13d
  size_t v17; // [rsp+8h] [rbp-48h]
  int v18; // [rsp+1Ch] [rbp-34h]

  v3 = *(_DWORD *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  if ( v3 )
  {
    v5 = v3 - 1;
    v6 = sub_C94890(*(_QWORD **)a2, *(_QWORD *)(a2 + 8));
    v7 = *(_QWORD *)(a2 + 8);
    v8 = *(const void **)a2;
    v9 = 1;
    for ( i = v5 & v6; ; i = v5 & v16 )
    {
      v11 = v4 + 24LL * i;
      v12 = *(const void **)v11;
      if ( *(_QWORD *)v11 == -1 )
        break;
      if ( v12 == (const void *)-2LL )
      {
        if ( v8 == (const void *)-2LL )
          goto LABEL_8;
      }
      else if ( v7 == *(_QWORD *)(v11 + 8) )
      {
        v18 = v9;
        if ( !v7 )
          goto LABEL_8;
        v17 = v7;
        v13 = memcmp(v8, v12, v7);
        v7 = v17;
        v9 = v18;
        if ( !v13 )
          goto LABEL_8;
      }
      v16 = v9 + i;
      ++v9;
    }
    if ( v8 != (const void *)-1LL )
      goto LABEL_11;
LABEL_8:
    v14 = *(_QWORD *)(a1 + 32);
    if ( v11 == *(_QWORD *)(a1 + 8) + 24LL * *(unsigned int *)(a1 + 24) )
      return v14 + 32LL * *(unsigned int *)(a1 + 40);
    return v14 + 32LL * *(unsigned int *)(v11 + 16);
  }
  else
  {
LABEL_11:
    v14 = *(_QWORD *)(a1 + 32);
    return v14 + 32LL * *(unsigned int *)(a1 + 40);
  }
}
