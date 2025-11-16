// Function: sub_161AC40
// Address: 0x161ac40
//
__int64 __fastcall sub_161AC40(__int64 a1, unsigned __int64 a2, __int64 a3, unsigned __int64 a4)
{
  __int64 v6; // r12
  unsigned __int64 v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v8[0] = a2;
  v6 = *(_QWORD *)sub_161A990(a1 + 568, v8);
  sub_160FA70(v6);
  sub_161A600(v6, a4);
  if ( v6 )
    v6 += 568;
  return sub_160EA80(v6, a3);
}
