// Function: sub_876ED0
// Address: 0x876ed0
//
_BYTE *__fastcall sub_876ED0(char *a1)
{
  _BYTE *result; // rax
  __int64 v2; // rsi
  char *v3; // rdx
  char v4; // cl
  __int64 v5; // r9
  __int64 v6; // r8
  char v7; // r10
  char v8; // cl

  if ( !dword_4D0460C || (result = (_BYTE *)qword_4F5FF90) == 0 || qword_4F5FF90 > (unsigned __int64)qword_4F5FF88 )
  {
    v2 = qword_4F5FF98;
    v3 = a1 + 1;
    v4 = *a1;
    result = (_BYTE *)(qword_4F5FF98 + 1);
    *(_BYTE *)qword_4F5FF98 = *a1;
    qword_4F5FF98 = (__int64)result;
    if ( v4 )
    {
      v5 = qword_4F5FF90;
      v6 = qword_4F5FF88;
      v7 = 0;
      while ( 1 )
      {
        if ( dword_4D0460C && v5 )
        {
          if ( v5 == ++v6 )
          {
            qword_4F5FF88 = v5;
            qword_4F5FF98 = (__int64)--result;
            return result;
          }
          v7 = 1;
        }
        v8 = *v3++;
        *result = v8;
        if ( !v8 )
          break;
        ++result;
      }
      if ( v7 )
        qword_4F5FF88 = v6;
    }
    else
    {
      result = (_BYTE *)v2;
    }
    qword_4F5FF98 = (__int64)result;
  }
  return result;
}
