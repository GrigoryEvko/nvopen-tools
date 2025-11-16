// Function: sub_134ACC0
// Address: 0x134acc0
//
int __fastcall sub_134ACC0(__int64 a1, __int64 a2)
{
  int result; // eax

  sub_1340B70(a1, a2 + 68096);
  sub_133DF80(a1, a2 + 80);
  sub_133DF80(a1, a2 + 19520);
  sub_133DF80(a1, a2 + 38960);
  sub_130B050(a1, a2 + 58432);
  sub_130B050(a1, a2 + 58672);
  result = sub_130B050(a1, a2 + 60456);
  if ( *(_BYTE *)(a2 + 17) )
  {
    sub_130EFF0(a1, a2 + 62264);
    return sub_1348960(a1, a2 + 62384);
  }
  return result;
}
