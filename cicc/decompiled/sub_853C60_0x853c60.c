// Function: sub_853C60
// Address: 0x853c60
//
_DWORD *__fastcall sub_853C60(int a1)
{
  _DWORD *result; // rax
  _QWORD *v3; // rbx
  __int64 v4; // rdi
  __int64 v5; // r13

  if ( unk_4F04C48 == -1
    || (result = (_DWORD *)(qword_4F04C68[0] + 776LL * unk_4F04C48), (*((_BYTE *)result + 10) & 1) == 0) )
  {
    result = &dword_4F04C44;
    if ( dword_4F04C44 == -1 )
    {
      result = &qword_4D03E88;
      v3 = (_QWORD *)qword_4D03E88;
      if ( qword_4D03E88 )
      {
        do
        {
          if ( !v3[8] )
          {
            if ( !a1 || (result = (_DWORD *)v3[1], result[3] == a1) )
            {
              result = (_DWORD *)sub_853BE0((__int64)v3);
              v4 = v3[11];
              v3[8] = result;
              if ( v4 )
              {
                v5 = *(_QWORD *)(v4 + 40);
                if ( !dword_4F04C3C )
                  result = (_DWORD *)sub_8699D0(v4, 58, result);
                if ( v5 )
                  result = (_DWORD *)sub_86A080(v5);
                v3[8] = 0;
              }
            }
          }
          v3 = (_QWORD *)*v3;
        }
        while ( v3 );
      }
    }
  }
  return result;
}
