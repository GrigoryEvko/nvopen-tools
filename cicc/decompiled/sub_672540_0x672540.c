// Function: sub_672540
// Address: 0x672540
//
__int64 __fastcall sub_672540(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // rdx
  __int64 v6; // rcx

  while ( 1 )
  {
    result = word_4F06418[0];
    if ( word_4F06418[0] != 142 )
      break;
    sub_7AE360(a1);
    result = sub_7B8B50(a1, a2, v5, v6);
    if ( word_4F06418[0] != 27 )
      return result;
LABEL_5:
    a2 = 0;
    sub_7C6040(a1, 0);
    sub_7AE360(a1);
    sub_7B8B50(a1, 0, v3, v4);
  }
  if ( word_4F06418[0] == 25 )
  {
    result = (unsigned int)dword_4D043F8;
    if ( dword_4D043F8 )
    {
      result = sub_7BE840(0, 0);
      if ( (_WORD)result == 25 )
        goto LABEL_5;
    }
  }
  return result;
}
