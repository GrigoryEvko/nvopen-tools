// Function: sub_733700
// Address: 0x733700
//
__int64 __fastcall sub_733700(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // rcx
  __int64 v6; // rdx

  result = *(unsigned __int8 *)(a1 + 24);
  if ( (_BYTE)result == 10 )
  {
    v3 = *(_QWORD *)(a1 + 64);
    v4 = *(_QWORD *)(v3 + 32);
    if ( *(_BYTE *)(v3 + 8) )
      sub_733650(*(_QWORD *)(a1 + 64));
    result = *(_QWORD *)(v4 + 48);
    v5 = *(_QWORD *)(v3 + 56);
    if ( v3 == result )
    {
      *(_QWORD *)(v4 + 48) = v5;
    }
    else
    {
      do
      {
        v6 = result;
        result = *(_QWORD *)(result + 56);
      }
      while ( v3 != result );
      *(_QWORD *)(v6 + 56) = v5;
    }
  }
  else if ( (_BYTE)result == 17 )
  {
    *(_DWORD *)(a2 + 76) = 1;
  }
  return result;
}
