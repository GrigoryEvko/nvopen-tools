// Function: sub_6510D0
// Address: 0x6510d0
//
_BOOL8 sub_6510D0()
{
  __int64 v0; // rdx
  __int64 v1; // rcx
  _BOOL4 v2; // r13d
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rdx
  __int64 v7; // rcx
  _BYTE v8[64]; // [rsp+0h] [rbp-40h] BYREF

  sub_7ADF70(v8, 0);
  sub_7AE360(v8);
  sub_7B8B50(v8, 0, v0, v1);
  if ( word_4F06418[0] == 27 && (sub_7AE360(v8), sub_7B8B50(v8, 0, v4, v5), word_4F06418[0] == 77) )
  {
    sub_7AE360(v8);
    sub_7B8B50(v8, 0, v6, v7);
    v2 = word_4F06418[0] == 28;
  }
  else
  {
    v2 = 0;
  }
  sub_7BC000(v8);
  return v2;
}
