// Function: sub_77A750
// Address: 0x77a750
//
unsigned __int64 __fastcall sub_77A750(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v4; // rbx
  int v5; // edi
  unsigned int v6; // edx
  unsigned __int64 result; // rax
  __int64 v8; // rcx
  __int64 v9; // r14
  unsigned __int64 v10; // rsi
  char i; // al
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rsi
  int v15; // edx
  __int64 v16; // rcx
  unsigned int v17; // ebx
  __int64 v18; // rdi
  unsigned int v19; // eax
  int v20[9]; // [rsp+Ch] [rbp-24h] BYREF

  v4 = a2 >> 3;
  v5 = *(_DWORD *)(a1 + 8);
  v6 = (a2 >> 3) & v5;
  result = *(_QWORD *)a1 + 16LL * v6;
  v8 = *(_QWORD *)result;
  if ( *(_QWORD *)result == a2 )
  {
LABEL_6:
    v9 = *(_QWORD *)(result + 8);
    if ( !v9 )
      return result;
    v10 = *(_QWORD *)(a2 + 120);
    v20[0] = 1;
    for ( i = *(_BYTE *)(v10 + 140); i == 12; i = *(_BYTE *)(v10 + 140) )
      v10 = *(_QWORD *)(v10 + 160);
    v12 = 16;
    if ( (unsigned __int8)(i - 2) > 1u )
    {
      v19 = sub_7764B0(a1, v10, v20);
      if ( (v19 & 7) != 0 )
      {
        v13 = v9 + v19 + 8 - (v19 & 7);
        result = *(unsigned int *)(*(_QWORD *)(a1 + 72) + 40LL);
        if ( *(_DWORD *)v13 < (unsigned int)result )
          return result;
        goto LABEL_11;
      }
      v12 = v19;
    }
    v13 = v9 + v12;
    result = *(unsigned int *)(*(_QWORD *)(a1 + 72) + 40LL);
    if ( *(_DWORD *)v13 < (unsigned int)result )
      return result;
LABEL_11:
    v14 = *(_QWORD *)(v13 + 8);
    v15 = *(_DWORD *)(a1 + 8);
    v16 = *(_QWORD *)a1;
    v17 = v15 & v4;
    result = *(_QWORD *)a1 + 16LL * v17;
    v18 = *(_QWORD *)result;
    if ( v14 )
    {
      if ( a2 != v18 )
      {
        do
        {
          v17 = v15 & (v17 + 1);
          result = v16 + 16LL * v17;
        }
        while ( a2 != *(_QWORD *)result );
      }
      *(_QWORD *)(result + 8) = v14;
    }
    else
    {
      if ( a2 != v18 )
      {
        do
        {
          v17 = v15 & (v17 + 1);
          result = v16 + 16LL * v17;
        }
        while ( *(_QWORD *)result != a2 );
      }
      *(_QWORD *)result = 0;
      result = v17 + 1;
      if ( *(_QWORD *)(v16 + 16LL * ((unsigned int)result & v15)) )
        result = sub_771200(*(_QWORD *)a1, *(_DWORD *)(a1 + 8), v17);
      --*(_DWORD *)(a1 + 12);
    }
    return result;
  }
  while ( v8 )
  {
    v6 = v5 & (v6 + 1);
    result = *(_QWORD *)a1 + 16LL * v6;
    v8 = *(_QWORD *)result;
    if ( a2 == *(_QWORD *)result )
      goto LABEL_6;
  }
  return result;
}
