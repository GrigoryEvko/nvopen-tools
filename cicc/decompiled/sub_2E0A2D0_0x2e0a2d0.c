// Function: sub_2E0A2D0
// Address: 0x2e0a2d0
//
__int64 __fastcall sub_2E0A2D0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx

  result = (unsigned int)(*(_DWORD *)(a1 + 72) - 1);
  if ( *(_DWORD *)a2 == (_DWORD)result )
  {
    v3 = 8LL * (unsigned int)result - 8;
    do
    {
      *(_DWORD *)(a1 + 72) = result;
      if ( !(_DWORD)result )
        break;
      result = (unsigned int)(result - 1);
      v4 = *(_QWORD *)(*(_QWORD *)(a1 + 64) + v3);
      v3 -= 8;
    }
    while ( (*(_QWORD *)(v4 + 8) & 0xFFFFFFFFFFFFFFF8LL) == 0 );
  }
  else
  {
    *(_QWORD *)(a2 + 8) = 0;
  }
  return result;
}
