// Function: sub_2210610
// Address: 0x2210610
//
__int64 __fastcall sub_2210610(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v5; // ebp
  __int64 v6; // rbx
  bool v7; // dl
  bool v8; // al
  _QWORD v10[4]; // [rsp+0h] [rbp-20h] BYREF

  v10[0] = a3;
  v10[1] = a4;
  if ( !a5 )
    return 0;
  v5 = a3;
  v6 = a5 - 1;
  do
  {
    v7 = (unsigned int)sub_220F920((__int64)v10, 0x10FFFFu) <= 0x10FFFF;
    v8 = v6-- != 0;
  }
  while ( v8 && v7 );
  return (unsigned int)(LODWORD(v10[0]) - v5);
}
