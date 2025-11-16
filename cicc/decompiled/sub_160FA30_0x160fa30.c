// Function: sub_160FA30
// Address: 0x160fa30
//
__int64 __fastcall sub_160FA30(__int64 a1)
{
  __int64 result; // rax
  _QWORD *v2; // rdx

  result = *(unsigned int *)(a1 + 192);
  if ( (_DWORD)result )
  {
    LODWORD(result) = 0;
    do
    {
      v2 = *(_QWORD **)(*(_QWORD *)(*(_QWORD *)(a1 + 184) + 8LL * (unsigned int)result) + 8LL);
      if ( v2[1] != *v2 )
        v2[1] = *v2;
      result = (unsigned int)(result + 1);
    }
    while ( *(_DWORD *)(a1 + 192) > (unsigned int)result );
  }
  return result;
}
