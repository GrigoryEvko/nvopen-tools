// Function: sub_262B840
// Address: 0x262b840
//
__int64 __fastcall sub_262B840(__int64 a1, __int64 a2)
{
  int v2; // r15d
  __int64 v3; // r14
  int v4; // r15d
  int v5; // eax
  const void *v6; // rcx
  size_t v7; // rbx
  int v8; // r8d
  unsigned int i; // r13d
  __int64 v10; // r12
  const void *v11; // rsi
  int v12; // eax
  unsigned int v14; // r13d
  const void *v15; // [rsp+0h] [rbp-40h]
  int v16; // [rsp+Ch] [rbp-34h]

  v2 = *(_DWORD *)(a1 + 24);
  if ( !v2 )
    return 0;
  v3 = *(_QWORD *)(a1 + 8);
  v4 = v2 - 1;
  v5 = sub_C94890(*(_QWORD **)a2, *(_QWORD *)(a2 + 8));
  v6 = *(const void **)a2;
  v7 = *(_QWORD *)(a2 + 8);
  v8 = 1;
  for ( i = v4 & v5; ; i = v4 & v14 )
  {
    v10 = v3 + 24LL * i;
    v11 = *(const void **)v10;
    if ( *(_QWORD *)v10 == -1 )
      break;
    if ( v11 == (const void *)-2LL )
    {
      if ( v6 == (const void *)-2LL )
        return v10;
    }
    else if ( v7 == *(_QWORD *)(v10 + 8) )
    {
      v16 = v8;
      if ( !v7 )
        return v10;
      v15 = v6;
      v12 = memcmp(v6, v11, v7);
      v6 = v15;
      v8 = v16;
      if ( !v12 )
        return v10;
    }
    v14 = v8 + i;
    ++v8;
  }
  if ( v6 != (const void *)-1LL )
    return 0;
  return v10;
}
