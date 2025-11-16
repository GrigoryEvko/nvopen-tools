// Function: sub_12BAF10
// Address: 0x12baf10
//
__int64 __fastcall sub_12BAF10(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  __int64 v4; // r15
  __int64 v5; // r13
  __int64 v6; // r13
  _QWORD v7[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = (unsigned __int8)byte_4F92D70;
  if ( byte_4F92D70 || !dword_4C6F008 )
  {
    if ( !qword_4F92D80 )
      sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
    v4 = qword_4F92D80;
    sub_16C30C0(qword_4F92D80);
    LOBYTE(v2) = a2 == 0 || a1 == 0;
    if ( (_BYTE)v2 )
    {
      v2 = 0;
    }
    else
    {
      sub_16C2450(v7, a1, a2, "<unnamed>", 9, 0);
      v6 = v7[0];
      if ( v7[0] )
      {
        v2 = sub_1C17C50(v7[0]);
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6);
      }
    }
    sub_16C30E0(v4);
  }
  else
  {
    if ( !qword_4F92D80 )
      sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
    if ( a1 != 0 && a2 != 0 )
    {
      v2 = 0;
      sub_16C2450(v7, a1, a2, "<unnamed>", 9, 0);
      v5 = v7[0];
      if ( v7[0] )
      {
        v2 = sub_1C17C50(v7[0]);
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v5 + 8LL))(v5);
      }
    }
  }
  return v2;
}
