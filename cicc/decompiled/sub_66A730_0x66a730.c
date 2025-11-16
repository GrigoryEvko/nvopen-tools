// Function: sub_66A730
// Address: 0x66a730
//
__int64 __fastcall sub_66A730(__int64 a1, _DWORD *a2, _DWORD *a3, _DWORD *a4)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 result; // rax

  if ( word_4F06418[0] == 241 )
  {
    sub_7B8B50(a1, a2, a3, a4);
    while ( word_4F06418[0] == 241 )
    {
      sub_684AC0(7, 1451);
      sub_7B8B50(7, 1451, v6, v7);
    }
    result = 1;
  }
  else
  {
    result = 0;
  }
  *a2 = result;
  *a3 = 0;
  *a4 = 0;
  return result;
}
