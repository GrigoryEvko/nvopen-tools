// Function: sub_7D3790
// Address: 0x7d3790
//
__int64 __fastcall sub_7D3790(unsigned __int8 a1, const char *a2)
{
  _QWORD v3[11]; // [rsp-58h] [rbp-58h] BYREF

  if ( !qword_4D04A60[a1] )
    return 0;
  sub_87A720(a1, v3, &dword_4F063F8);
  if ( sub_7D2AC0(v3, a2, 0x10u) )
    return v3[3];
  else
    return 0;
}
