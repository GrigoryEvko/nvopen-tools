// Function: sub_1993F10
// Address: 0x1993f10
//
__int64 __fastcall sub_1993F10(__int64 a1)
{
  int v1; // edx
  __int64 result; // rax
  __int64 v3; // rax
  _QWORD *v4; // rbx
  __int64 v5; // r12

  while ( 1 )
  {
    v1 = *(unsigned __int16 *)(a1 + 24);
    if ( (_WORD)v1 == 7 )
      return 1;
    result = (unsigned int)(v1 - 7);
    LOBYTE(result) = (unsigned int)(unsigned __int16)v1 - 4 <= 1 || (unsigned __int16)(v1 - 7) <= 2u;
    if ( (_BYTE)result )
      break;
    if ( (unsigned __int16)(v1 - 1) > 2u )
      return result;
    a1 = *(_QWORD *)(a1 + 32);
  }
  v3 = *(_QWORD *)(a1 + 40);
  if ( !(_DWORD)v3 )
    return 0;
  v4 = *(_QWORD **)(a1 + 32);
  v5 = (__int64)&v4[(unsigned int)(v3 - 1) + 1];
  do
  {
    result = sub_1993F10(*v4);
    if ( (_BYTE)result )
      break;
    ++v4;
  }
  while ( (_QWORD *)v5 != v4 );
  return result;
}
