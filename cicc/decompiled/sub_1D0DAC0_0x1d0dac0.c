// Function: sub_1D0DAC0
// Address: 0x1d0dac0
//
__int64 __fastcall sub_1D0DAC0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax

  a1[77] = a3;
  a1[78] = a2;
  sub_1F013F0();
  v3 = a1[80];
  if ( v3 != a1[81] )
    a1[81] = v3;
  return (*(__int64 (__fastcall **)(_QWORD *))(*a1 + 80LL))(a1);
}
