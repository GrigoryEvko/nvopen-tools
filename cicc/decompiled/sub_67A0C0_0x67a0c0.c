// Function: sub_67A0C0
// Address: 0x67a0c0
//
unsigned __int64 __fastcall sub_67A0C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int16 v4; // r14
  unsigned __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rdx

  v4 = word_4F06418[0];
  result = sub_7B8B50(a1, a2, a3, a4);
  if ( word_4F06418[0] == 27 )
  {
    sub_679930((unsigned __int16)a2, 0, v6, v7);
    if ( v4 == 189 && dword_4F077BC && qword_4F077A8 > 0x76BFu && !(unsigned int)sub_679C10(3u) )
    {
      sub_6797D0(0x1Cu, 1);
      result = sub_7BE840(0, 0);
      if ( (unsigned __int16)result <= 0x20u )
      {
        v8 = 0x1EA000000LL;
        if ( _bittest64(&v8, result) )
        {
          *(_QWORD *)(a1 + 12) = 0x100000001LL;
          return 0x100000001LL;
        }
      }
    }
    else
    {
      return sub_6797D0(0x1Cu, 1);
    }
  }
  return result;
}
