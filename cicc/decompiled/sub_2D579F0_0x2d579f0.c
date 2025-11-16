// Function: sub_2D579F0
// Address: 0x2d579f0
//
__int64 *__fastcall sub_2D579F0(__int64 a1, unsigned __int64 *a2)
{
  __int64 *result; // rax
  unsigned __int64 v4; // rsi
  __int64 v5; // r10
  int v6; // ebx
  unsigned int v7; // ecx
  __int64 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // rdx
  char v11; // r11
  __int64 v12; // rax
  __int64 v13; // rax
  int v14; // eax
  int v15; // r13d

  result = (__int64 *)*(unsigned int *)(a1 + 8);
  v4 = *a2;
  if ( v4 < (unsigned __int64)result )
  {
    while ( 1 )
    {
      v10 = *(_QWORD *)(*(_QWORD *)a1 + 8 * v4);
      v11 = *(_BYTE *)(a1 + 280) & 1;
      if ( v11 )
      {
        v5 = a1 + 288;
        v6 = 31;
      }
      else
      {
        v12 = *(unsigned int *)(a1 + 296);
        v5 = *(_QWORD *)(a1 + 288);
        if ( !(_DWORD)v12 )
          goto LABEL_14;
        v6 = v12 - 1;
      }
      v7 = v6 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      result = (__int64 *)(v5 + 16LL * v7);
      v8 = *result;
      if ( v10 != *result )
        break;
LABEL_5:
      v9 = 512;
      if ( !v11 )
        v9 = 16LL * *(unsigned int *)(a1 + 296);
      if ( result == (__int64 *)(v5 + v9) || result[1] != v4 )
      {
        *a2 = ++v4;
        result = (__int64 *)*(unsigned int *)(a1 + 8);
        if ( v4 < (unsigned __int64)result )
          continue;
      }
      return result;
    }
    v14 = 1;
    while ( v8 != -4096 )
    {
      v15 = v14 + 1;
      v7 = v6 & (v14 + v7);
      result = (__int64 *)(v5 + 16LL * v7);
      v8 = *result;
      if ( v10 == *result )
        goto LABEL_5;
      v14 = v15;
    }
    if ( v11 )
    {
      v13 = 512;
    }
    else
    {
      v12 = *(unsigned int *)(a1 + 296);
LABEL_14:
      v13 = 16 * v12;
    }
    result = (__int64 *)(v5 + v13);
    goto LABEL_5;
  }
  return result;
}
