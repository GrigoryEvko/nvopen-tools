// Function: sub_7A65D0
// Address: 0x7a65d0
//
__int64 __fastcall sub_7A65D0(_DWORD *a1, __int64 a2)
{
  __int64 result; // rax

  if ( !HIDWORD(qword_4F077B4)
    || unk_4D04250 > 0x765Bu
    || unk_4F06AB0 >= 0
    || (result = sub_8D3B10(a2), !(_DWORD)result) )
  {
    result = *(unsigned int *)(a2 + 184);
    if ( (_DWORD)result )
    {
      if ( *a1 > (unsigned int)result )
        *a1 = result;
    }
  }
  return result;
}
