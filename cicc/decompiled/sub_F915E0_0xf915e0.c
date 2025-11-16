// Function: sub_F915E0
// Address: 0xf915e0
//
bool __fastcall sub_F915E0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _QWORD *a7,
        int a8,
        __int64 a9)
{
  __int64 v10; // rbx
  __int64 v11; // rdi
  int v12; // edx
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // r8
  __int64 v17; // rax
  __int64 v18; // r9
  _QWORD *v19; // rdx
  _QWORD *v20; // rcx
  _QWORD *v21; // rax
  __int64 v22; // rax
  __int64 *v24; // rax
  __int64 v25; // [rsp+8h] [rbp-38h]

  v10 = a1;
  if ( a2 == a1 )
    return a2 == v10;
  while ( 1 )
  {
    v11 = *(_QWORD *)(v10 - 8);
    v12 = *(_DWORD *)(v10 + 4) & 0x7FFFFFF;
    if ( v12 )
    {
      v13 = 0;
      v14 = v11 + 32LL * *(unsigned int *)(v10 + 72);
      do
      {
        if ( *a7 == *(_QWORD *)(v14 + 8 * v13) )
        {
          v15 = a7[1];
          v16 = *(_QWORD *)(v11 + 32 * v13);
          goto LABEL_7;
        }
        ++v13;
      }
      while ( v12 != (_DWORD)v13 );
      v16 = *(_QWORD *)(v11 + 0x1FFFFFFFE0LL);
      v15 = a7[1];
LABEL_7:
      v17 = 0;
      do
      {
        if ( *(_QWORD *)(v14 + 8 * v17) == v15 )
        {
          v18 = *(_QWORD *)(v11 + 32 * v17);
          goto LABEL_11;
        }
        ++v17;
      }
      while ( v12 != (_DWORD)v17 );
      v18 = *(_QWORD *)(v11 + 0x1FFFFFFFE0LL);
LABEL_11:
      if ( v18 != v16 )
        break;
    }
LABEL_21:
    v22 = *(_QWORD *)(v10 + 32);
    if ( !v22 )
      BUG();
    v10 = 0;
    if ( *(_BYTE *)(v22 - 24) == 84 )
      v10 = v22 - 24;
    if ( a2 == v10 )
      return a2 == v10;
  }
  if ( !a9 )
    return a2 == v10;
  if ( !*(_BYTE *)(a9 + 28) )
  {
    v25 = v18;
    v24 = sub_C8CA60(a9, v16);
    v18 = v25;
    if ( !v24 )
      return a2 == v10;
    if ( *(_BYTE *)(a9 + 28) )
    {
      v19 = *(_QWORD **)(a9 + 8);
      v20 = &v19[*(unsigned int *)(a9 + 20)];
      if ( v19 == v20 )
        return a2 == v10;
LABEL_20:
      while ( *v19 != v18 )
      {
        if ( v20 == ++v19 )
          return a2 == v10;
      }
    }
    else if ( !sub_C8CA60(a9, v25) )
    {
      return a2 == v10;
    }
    goto LABEL_21;
  }
  v19 = *(_QWORD **)(a9 + 8);
  v20 = &v19[*(unsigned int *)(a9 + 20)];
  if ( v19 != v20 )
  {
    v21 = *(_QWORD **)(a9 + 8);
    do
    {
      if ( *v21 == v16 )
        goto LABEL_20;
      ++v21;
    }
    while ( v20 != v21 );
  }
  return a2 == v10;
}
