// Function: sub_22101B0
// Address: 0x22101b0
//
__int64 __fastcall sub_22101B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5)
{
  unsigned __int64 v7; // rbx
  unsigned __int64 v8; // rbp
  unsigned int v9; // eax
  _QWORD v11[6]; // [rsp+0h] [rbp-30h] BYREF

  v7 = 1;
  v11[0] = a3;
  v11[1] = a4;
  if ( a5 <= 1 )
  {
LABEL_8:
    if ( a5 == v7 )
      sub_220FAE0((__int64)v11, 0xFFFFu);
  }
  else
  {
    v8 = 0;
    while ( 1 )
    {
      v9 = sub_220FAE0((__int64)v11, 0x10FFFFu);
      if ( v9 > 0x10FFFF )
        break;
      if ( v9 <= 0xFFFF )
        v7 = v8;
      v8 = v7 + 1;
      v7 += 2LL;
      if ( a5 <= v7 )
        goto LABEL_8;
    }
  }
  return v11[0] - a3;
}
