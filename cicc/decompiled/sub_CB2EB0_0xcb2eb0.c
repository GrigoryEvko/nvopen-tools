// Function: sub_CB2EB0
// Address: 0xcb2eb0
//
const char *__fastcall sub_CB2EB0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int64 v6; // [rsp+8h] [rbp-18h] BYREF

  if ( sub_C93CC0(a1, a2, 0, &v6) )
    return "invalid number";
  *a4 = v6;
  return 0;
}
