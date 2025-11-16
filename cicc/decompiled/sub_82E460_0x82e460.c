// Function: sub_82E460
// Address: 0x82e460
//
__int64 __fastcall sub_82E460(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r12

  v2 = a1;
  if ( (unsigned int)sub_8D2E30(a1) )
    v2 = sub_8D46C0(a1);
  sub_7461E0((__int64)&qword_4F5F780);
  qword_4F5F780 = (__int64)sub_729610;
  byte_4F5F811 = dword_4F07460;
  qword_4F06C40 = 0;
  sub_74B930(v2, (__int64)&qword_4F5F780);
  sub_729660(0);
  return sub_67DCF0(a2, 1085, (__int64)qword_4F06C50);
}
