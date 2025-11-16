// Function: sub_1341180
// Address: 0x1341180
//
__int64 __fastcall sub_1341180(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rdx
  __int64 v5; // rdi
  __int64 v7; // rax

  v4 = a4;
  if ( !unk_4F96B58 )
    return sub_1341160(0, a2, a4);
  v5 = __readfsqword(0) - 2664;
  if ( __readfsbyte(0xFFFFF8C8) )
  {
    v7 = sub_1313D30(v5, 0);
    v4 = a4;
    v5 = v7;
  }
  return sub_1341160(v5, a2, v4);
}
