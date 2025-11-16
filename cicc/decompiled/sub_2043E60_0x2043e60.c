// Function: sub_2043E60
// Address: 0x2043e60
//
__int64 __fastcall sub_2043E60(__int64 a1)
{
  int v1; // eax
  __int64 result; // rax
  int v3; // eax
  _QWORD *v4; // r12
  __int64 v5; // rbx

  v1 = *(_DWORD *)(a1 + 60);
  if ( !v1 )
    return 0;
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 40) + 16LL * (unsigned int)(v1 - 1)) == 1 )
    return a1;
  v3 = *(_DWORD *)(a1 + 56);
  if ( !v3 )
    return 0;
  v4 = *(_QWORD **)(a1 + 32);
  v5 = (__int64)&v4[5 * (unsigned int)(v3 - 1) + 5];
  while ( 1 )
  {
    result = sub_2043E60(*v4);
    if ( result )
      break;
    v4 += 5;
    if ( v4 == (_QWORD *)v5 )
      return 0;
  }
  return result;
}
