// Function: sub_697260
// Address: 0x697260
//
_QWORD *__fastcall sub_697260(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *result; // rax
  char v5; // dl

  result = (_QWORD *)sub_8D2600(a1);
  if ( !(_DWORD)result )
  {
    v5 = *(_BYTE *)(a1 + 140);
    if ( v5 == 12 )
    {
      result = (_QWORD *)a1;
      do
      {
        result = (_QWORD *)result[20];
        v5 = *((_BYTE *)result + 140);
      }
      while ( v5 == 12 );
    }
    if ( v5 )
    {
      result = (_QWORD *)sub_8D3410(a1);
      if ( !(_DWORD)result )
      {
        result = (_QWORD *)sub_8DBE70(a1);
        if ( !(_DWORD)result )
        {
          if ( dword_4F077BC )
          {
            if ( !(_DWORD)qword_4F077B4 )
            {
              result = &qword_4F077A8;
              if ( qword_4F077A8 <= 0x1ADAFu )
                return result;
            }
          }
          else if ( !(_DWORD)qword_4F077B4 )
          {
            goto LABEL_13;
          }
          result = (_QWORD *)sub_696840(a2);
          if ( !(_DWORD)result )
          {
LABEL_13:
            sub_6E5F60(a3, a1, 8);
            return (_QWORD *)sub_6E6260(a2);
          }
        }
      }
    }
  }
  return result;
}
