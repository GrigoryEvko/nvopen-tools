// Function: sub_3373BA0
// Address: 0x3373ba0
//
__int64 __fastcall sub_3373BA0(__int64 a1)
{
  __int64 v1; // rax
  __int64 result; // rax

  *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 960) + 744LL) + 233LL) = 1;
  v1 = sub_B2E500(**(_QWORD **)(a1 + 960));
  result = sub_B2A630(v1);
  if ( (_DWORD)result != 12 )
  {
    *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 960) + 744LL) + 235LL) = 1;
    result = *(_QWORD *)(*(_QWORD *)(a1 + 960) + 744LL);
    *(_BYTE *)(result + 236) = 1;
  }
  return result;
}
