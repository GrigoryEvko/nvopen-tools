// Function: sub_131D250
// Address: 0x131d250
//
__int64 __fastcall sub_131D250(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 16);
  if ( result )
  {
    *(_BYTE *)(result + *(_QWORD *)(a1 + 32)) = 0;
    result = (*(__int64 (__fastcall **)(_QWORD, _QWORD))a1)(*(_QWORD *)(a1 + 8), *(_QWORD *)(a1 + 16));
    *(_QWORD *)(a1 + 32) = 0;
  }
  return result;
}
