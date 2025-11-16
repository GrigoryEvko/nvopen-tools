// Function: sub_BCADB0
// Address: 0xbcadb0
//
__int64 __fastcall sub_BCADB0(__int64 a1)
{
  char v1; // dl
  __int64 result; // rax
  int v3; // eax
  _QWORD *v4; // rbx
  __int64 v5; // r12

  while ( 1 )
  {
    v1 = *(_BYTE *)(a1 + 8);
    if ( v1 != 16 )
      break;
    if ( !*(_DWORD *)(a1 + 32) )
      return 1;
    a1 = *(_QWORD *)(a1 + 24);
  }
  result = 0;
  if ( v1 == 15 )
  {
    v3 = *(_DWORD *)(a1 + 12);
    if ( !v3 )
      return 1;
    v4 = *(_QWORD **)(a1 + 16);
    v5 = (__int64)&v4[(unsigned int)(v3 - 1) + 1];
    do
    {
      result = sub_BCADB0(*v4);
      if ( !(_BYTE)result )
        break;
      ++v4;
    }
    while ( v4 != (_QWORD *)v5 );
  }
  return result;
}
