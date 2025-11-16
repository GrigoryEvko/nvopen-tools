// Function: sub_13086C0
// Address: 0x13086c0
//
__int64 __fastcall sub_13086C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rcx
  unsigned __int64 v5; // rdi
  char v7; // al
  __int64 v8; // rax
  __int64 v10; // [rsp+8h] [rbp-18h]

  v3 = a3;
  if ( dword_4C6F034[0] )
  {
    v7 = sub_13022D0(a1, a2);
    v3 = a3;
    if ( v7 )
      return 11;
  }
  if ( __readfsbyte(0xFFFFF8C8) )
  {
    v10 = v3;
    v8 = sub_1313D30(__readfsqword(0) - 2664, 0);
    v3 = v10;
    v5 = v8;
  }
  else
  {
    v5 = __readfsqword(0) - 2664;
  }
  return sub_133D460(v5, a1, a2, v3);
}
