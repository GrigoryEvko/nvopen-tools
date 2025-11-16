// Function: sub_12DDDE0
// Address: 0x12ddde0
//
__int64 __fastcall sub_12DDDE0(__int64 a1, _QWORD *a2, _QWORD *a3)
{
  int v4; // r12d
  __int64 v6; // r14
  int v7; // r12d
  int v8; // eax
  const void *v9; // rcx
  size_t v10; // rdx
  __int64 v11; // r8
  int v12; // r9d
  unsigned int i; // r15d
  __int64 v14; // rbx
  const void *v15; // rsi
  unsigned int v16; // r15d
  int v17; // [rsp+4h] [rbp-4Ch]
  __int64 v18; // [rsp+8h] [rbp-48h]
  size_t v19; // [rsp+10h] [rbp-40h]
  const void *v20; // [rsp+18h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *(_QWORD *)(a1 + 8);
    v7 = v4 - 1;
    v8 = sub_16D3930(*a2, a2[1]);
    v9 = (const void *)*a2;
    v10 = a2[1];
    v11 = 0;
    v12 = 1;
    for ( i = v7 & v8; ; i = v7 & v16 )
    {
      v14 = v6 + 16LL * i;
      v15 = *(const void **)v14;
      if ( *(_QWORD *)v14 == -1 )
        break;
      if ( v15 == (const void *)-2LL )
      {
        if ( v9 == (const void *)-2LL )
          goto LABEL_9;
        if ( !v11 )
          v11 = v6 + 16LL * i;
      }
      else if ( *(_QWORD *)(v14 + 8) == v10 )
      {
        if ( !v10 )
          goto LABEL_9;
        v17 = v12;
        v18 = v11;
        v19 = v10;
        v20 = v9;
        if ( !memcmp(v9, v15, v10) )
          goto LABEL_9;
        v9 = v20;
        v10 = v19;
        v11 = v18;
        v12 = v17;
      }
      v16 = v12 + i;
      ++v12;
    }
    if ( v9 == (const void *)-1LL )
    {
LABEL_9:
      *a3 = v14;
      return 1;
    }
    if ( !v11 )
      v11 = v6 + 16LL * i;
    *a3 = v11;
    return 0;
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
