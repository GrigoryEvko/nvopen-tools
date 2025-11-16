// Function: sub_18B4850
// Address: 0x18b4850
//
__int64 __fastcall sub_18B4850(__int64 a1, __int64 *a2, __int64 **a3)
{
  __int64 result; // rax
  int v5; // edx
  __int64 *v6; // r11
  __int64 v7; // r9
  int v8; // ebx
  __int64 v9; // rdi
  __int64 v10; // rcx
  unsigned int i; // eax
  __int64 *v12; // r10
  __int64 v13; // rsi
  unsigned int v14; // eax

  result = *(unsigned int *)(a1 + 24);
  if ( !(_DWORD)result )
  {
    *a3 = 0;
    return result;
  }
  v5 = result - 1;
  v6 = 0;
  v7 = *(_QWORD *)(a1 + 8);
  v8 = 1;
  v9 = *a2;
  v10 = a2[1];
  for ( i = (result - 1) & ((37 * v10) ^ ((unsigned int)*a2 >> 4) ^ ((unsigned int)*a2 >> 9)); ; i = v5 & v14 )
  {
    v12 = (__int64 *)(v7 + 24LL * i);
    v13 = *v12;
    if ( v9 == *v12 && v10 == v12[1] )
    {
      *a3 = v12;
      return 1;
    }
    if ( v13 == -4 )
      break;
    if ( v13 == -8 && v12[1] == -2 && !v6 )
      v6 = (__int64 *)(v7 + 24LL * i);
LABEL_6:
    v14 = v8 + i;
    ++v8;
  }
  if ( v12[1] != -1 )
    goto LABEL_6;
  if ( !v6 )
    v6 = (__int64 *)(v7 + 24LL * i);
  *a3 = v6;
  return 0;
}
