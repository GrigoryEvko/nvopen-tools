// Function: sub_8D6010
// Address: 0x8d6010
//
__int64 __fastcall sub_8D6010(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  unsigned int v3; // edx
  _DWORD *v4; // rdx

  result = a2;
  if ( *(char *)(a1 + 142) < 0 )
  {
    v3 = *(_DWORD *)(a1 + 136);
    if ( v3 >= a2 )
    {
      return v3;
    }
    else
    {
      v4 = (_DWORD *)(a1 + 64);
      if ( HIDWORD(qword_4F077B4) && !(_DWORD)qword_4F077B4 )
      {
        sub_684AA0(5u, 0x759u, v4);
        return a2;
      }
      else
      {
        sub_684AA0(7u, 0x759u, v4);
        return a2;
      }
    }
  }
  return result;
}
