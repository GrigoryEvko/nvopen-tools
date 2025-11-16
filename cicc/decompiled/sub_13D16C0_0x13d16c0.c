// Function: sub_13D16C0
// Address: 0x13d16c0
//
__int64 __fastcall sub_13D16C0(__int64 a1, const void *a2, unsigned int a3)
{
  __int64 v4; // r12
  unsigned __int8 v5; // al
  __int64 v7; // r14
  unsigned int v8; // ebx
  __int64 v9; // rdx
  size_t v10; // rdx

  v4 = a1;
  v5 = *(_BYTE *)(a1 + 16);
  if ( v5 <= 0x10u )
    return sub_1584BD0();
  if ( v5 == 87 )
  {
    v7 = a3;
    while ( 1 )
    {
      v8 = *(_DWORD *)(v4 + 64);
      v9 = v8;
      if ( (unsigned int)v7 <= v8 )
        v9 = v7;
      v10 = 4 * v9;
      if ( !v10 || !memcmp(*(const void **)(v4 + 56), a2, v10) )
        break;
      v4 = *(_QWORD *)(v4 - 48);
      if ( !v4 )
        BUG();
      if ( *(_BYTE *)(v4 + 16) != 87 )
        return 0;
    }
    if ( a3 == v8 )
      return *(_QWORD *)(v4 - 24);
  }
  return 0;
}
