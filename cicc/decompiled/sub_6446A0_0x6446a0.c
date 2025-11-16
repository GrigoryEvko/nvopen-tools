// Function: sub_6446A0
// Address: 0x6446a0
//
__int64 __fastcall sub_6446A0(__int64 *a1, unsigned __int8 a2)
{
  __int64 result; // rax
  __int64 v4; // rdi
  char v5; // dl

  result = *a1;
  if ( *a1 )
  {
    if ( a2 )
    {
      if ( a2 == 3 )
      {
LABEL_6:
        *a1 = 0;
        return result;
      }
      v4 = a2;
    }
    else if ( (_DWORD)qword_4F077B4 )
    {
      v5 = *(_BYTE *)(result + 9);
      if ( v5 != 1 && v5 != 4 )
        goto LABEL_6;
      v4 = 7;
    }
    else
    {
      v4 = HIDWORD(qword_4F077B4) == 0 ? 7 : 5;
    }
    result = sub_684AA0(v4, 1847, *(_QWORD *)(result + 40));
    goto LABEL_6;
  }
  return result;
}
