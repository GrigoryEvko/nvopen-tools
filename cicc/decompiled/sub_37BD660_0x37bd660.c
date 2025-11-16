// Function: sub_37BD660
// Address: 0x37bd660
//
__int64 __fastcall sub_37BD660(__int64 a1, unsigned __int16 *a2, __int16 **a3)
{
  int v4; // edx
  int v5; // edx
  __int16 *v6; // r11
  int v7; // ebx
  __int64 v8; // r8
  unsigned __int16 v9; // di
  __int16 v10; // r9
  unsigned __int64 v11; // rax
  unsigned int i; // eax
  __int16 *v13; // rsi
  __int16 v14; // r10
  unsigned int v15; // eax

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v5 = v4 - 1;
  v6 = 0;
  v7 = 1;
  v8 = *(_QWORD *)(a1 + 8);
  v9 = a2[1];
  v10 = *a2;
  v11 = 0xBF58476D1CE4E5B9LL * ((37 * (unsigned int)v9) | ((unsigned __int64)(37 * (unsigned int)*a2) << 32));
  for ( i = v5 & ((v11 >> 31) ^ v11); ; i = v5 & v15 )
  {
    v13 = (__int16 *)(v8 + 8LL * i);
    v14 = *v13;
    if ( v10 == *v13 && v9 == v13[1] )
    {
      *a3 = v13;
      return 1;
    }
    if ( v14 == -1 )
      break;
    if ( v14 == -2 && v13[1] == -2 && !v6 )
      v6 = (__int16 *)(v8 + 8LL * i);
LABEL_9:
    v15 = v7 + i;
    ++v7;
  }
  if ( v13[1] != -1 )
    goto LABEL_9;
  if ( !v6 )
    v6 = (__int16 *)(v8 + 8LL * i);
  *a3 = v6;
  return 0;
}
