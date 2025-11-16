// Function: sub_1340FA0
// Address: 0x1340fa0
//
__int64 __fastcall sub_1340FA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  unsigned __int64 v7; // rcx
  __int64 v8; // rdi
  __int64 v10; // rax
  __int64 v11; // [rsp+0h] [rbp-30h]
  __int64 v12; // [rsp+8h] [rbp-28h]
  __int64 v13; // [rsp+10h] [rbp-20h]
  unsigned __int64 v14; // [rsp+18h] [rbp-18h]

  v7 = (a4 + 4095) & 0xFFFFFFFFFFFFF000LL;
  if ( unk_4F96B58 )
  {
    v8 = __readfsqword(0) - 2664;
    if ( __readfsbyte(0xFFFFF8C8) )
    {
      v11 = a6;
      v12 = a5;
      v13 = a3;
      v14 = v7;
      v10 = sub_1313D30(v8, 0);
      v7 = v14;
      a3 = v13;
      a5 = v12;
      a6 = v11;
      v8 = v10;
    }
  }
  else
  {
    v8 = 0;
  }
  return sub_1340EA0(v8, a2, a3, v7, a5, a6, a7);
}
