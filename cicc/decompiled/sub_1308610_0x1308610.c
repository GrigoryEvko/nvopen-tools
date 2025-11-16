// Function: sub_1308610
// Address: 0x1308610
//
__int64 __fastcall sub_1308610(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r9
  unsigned __int64 v9; // rdi
  char v11; // al
  __int64 v12; // rax
  __int64 v14; // [rsp+8h] [rbp-28h]

  v5 = a5;
  if ( dword_4C6F034[0] )
  {
    v11 = sub_13022D0(a1, a2);
    v5 = a5;
    if ( v11 )
      return 11;
  }
  if ( __readfsbyte(0xFFFFF8C8) )
  {
    v14 = v5;
    v12 = sub_1313D30(__readfsqword(0) - 2664, 0);
    v5 = v14;
    v9 = v12;
  }
  else
  {
    v9 = __readfsqword(0) - 2664;
  }
  return sub_133D380(v9, a1, a2, a3, a4, v5);
}
