// Function: sub_1308750
// Address: 0x1308750
//
__int64 __fastcall sub_1308750(__int64 a1, int a2, int a3, int a4, int a5, __int64 a6)
{
  int v8; // r12d
  unsigned __int64 v10; // rdi
  __int64 result; // rax
  char v12; // al
  char v13; // r8
  int v14; // eax
  __int64 v15; // [rsp+8h] [rbp-38h]
  __int64 v16; // [rsp+8h] [rbp-38h]

  v8 = a1;
  if ( !dword_4C6F034[0] || (v15 = a6, v12 = sub_13022D0(a1, dword_4C6F034[0]), a6 = v15, v13 = v12, result = 11, !v13) )
  {
    if ( __readfsbyte(0xFFFFF8C8) )
    {
      v16 = a6;
      v14 = sub_1313D30(__readfsqword(0) - 2664, 0);
      a6 = v16;
      LODWORD(v10) = v14;
    }
    else
    {
      v10 = __readfsqword(0) - 2664;
    }
    return sub_133D4C0(v10, v8, a2, a3, a4, a5, a6);
  }
  return result;
}
