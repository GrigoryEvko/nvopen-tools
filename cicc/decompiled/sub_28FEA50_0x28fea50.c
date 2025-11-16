// Function: sub_28FEA50
// Address: 0x28fea50
//
__int64 __fastcall sub_28FEA50(_QWORD *a1, _QWORD *a2)
{
  __int64 result; // rax

  result = *a2 < *a1;
  if ( *a2 > *a1 )
    return 0xFFFFFFFFLL;
  return result;
}
