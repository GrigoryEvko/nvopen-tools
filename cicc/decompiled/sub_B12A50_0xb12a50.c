// Function: sub_B12A50
// Address: 0xb12a50
//
__int64 __fastcall sub_B12A50(__int64 a1, unsigned int a2)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 40);
  if ( result )
  {
    if ( *(_BYTE *)result == 4 )
    {
      return *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(result + 136) + 8LL * a2) + 136LL);
    }
    else if ( (unsigned __int8)(*(_BYTE *)result - 5) <= 0x1Fu )
    {
      return 0;
    }
    else
    {
      return *(_QWORD *)(result + 136);
    }
  }
  return result;
}
