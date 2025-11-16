// Function: sub_2566C40
// Address: 0x2566c40
//
_QWORD *__fastcall sub_2566C40(__int64 a1, __int64 *a2)
{
  __int64 v3; // rdi
  int v4; // edx
  int v5; // edx
  __int64 v6; // rcx
  int v7; // ebx
  __int64 v8; // rsi
  unsigned int i; // eax
  _QWORD *v10; // r8
  unsigned int v11; // eax

  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
    return 0;
  v5 = v4 - 1;
  v6 = *a2;
  v7 = 1;
  v8 = a2[1];
  for ( i = v5
          & (((unsigned int)v8 >> 9)
           ^ ((unsigned int)v8 >> 4)
           ^ (16 * (((unsigned int)v6 >> 4) ^ ((unsigned int)v6 >> 9)))); ; i = v5 & v11 )
  {
    v10 = (_QWORD *)(v3 + ((unsigned __int64)i << 6));
    if ( v6 == *v10 && v8 == v10[1] )
      break;
    if ( unk_4FEE4D0 == *v10 && unk_4FEE4D8 == v10[1] )
      return 0;
    v11 = v7 + i;
    ++v7;
  }
  return v10;
}
