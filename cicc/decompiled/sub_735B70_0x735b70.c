// Function: sub_735B70
// Address: 0x735b70
//
__int64 __fastcall sub_735B70(__int64 a1)
{
  __int64 result; // rax

  for ( result = a1; (*(_BYTE *)(result + 124) & 1) != 0; result = *(_QWORD *)(result + 128) )
    ;
  return result;
}
