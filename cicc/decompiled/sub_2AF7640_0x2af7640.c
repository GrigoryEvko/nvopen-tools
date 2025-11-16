// Function: sub_2AF7640
// Address: 0x2af7640
//
__int64 __fastcall sub_2AF7640(__int64 a1)
{
  unsigned __int8 v1; // cl
  __int64 result; // rax
  unsigned __int64 v3; // rdx
  int v4; // r12d
  __int64 v5; // r13
  int v6; // ebx
  unsigned int v7; // edx
  _QWORD *v8; // rbx
  __int64 v9; // r12

  v1 = *(_BYTE *)(a1 + 8);
  result = 1;
  if ( v1 <= 3u || v1 == 5 || v1 > 0x14u )
    return result;
  result = ((0x165450uLL >> v1) & 1) == 0;
  if ( ((0x165450uLL >> v1) & 1) != 0 )
    return 1;
  if ( v1 == 16 )
  {
    v3 = *(_QWORD *)(a1 + 32);
    if ( v3 <= (unsigned int)dword_500F028 )
    {
      v4 = *(_QWORD *)(a1 + 32);
      if ( v3 )
      {
        v5 = *(_QWORD *)(a1 + 24);
        v6 = 0;
        do
        {
          result = sub_2AF7640(v5);
          if ( !(_BYTE)result )
            break;
          ++v6;
        }
        while ( v4 != v6 );
      }
      return result;
    }
    return 0;
  }
  if ( v1 != 15 )
    return result;
  v7 = *(_DWORD *)(a1 + 12);
  if ( v7 > dword_500F028 )
    return 0;
  if ( v7 )
  {
    v8 = *(_QWORD **)(a1 + 16);
    v9 = (__int64)&v8[v7];
    do
    {
      result = sub_2AF7640(*v8);
      if ( !(_BYTE)result )
        break;
      ++v8;
    }
    while ( (_QWORD *)v9 != v8 );
  }
  return result;
}
