// Function: sub_6943F0
// Address: 0x6943f0
//
__int64 __fastcall sub_6943F0(int a1, __int64 a2)
{
  int v2; // ebx
  __int64 v3; // r15
  char *v4; // rax
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // rax

  v2 = a2;
  v3 = unk_4F06460;
  if ( sub_6877C0() )
  {
    a2 = 1;
    v4 = sub_693F60((const char *)word_4F06418[0], 1);
  }
  else
  {
    if ( !HIDWORD(qword_4F077B4) )
    {
      v9 = sub_693F30(word_4F06418[0]);
      a2 = (__int64)&dword_4F063F8;
      sub_6851A0(0x40Cu, &dword_4F063F8, v9);
    }
    v4 = "\"";
  }
  unk_4F06460 = v4;
  result = sub_7B78D0(v2 | 0x20u);
  unk_4F06460 = v3;
  word_4F06418[0] = result;
  if ( a1 )
    return sub_7B8270(1, a2, v6, v7, v8);
  return result;
}
