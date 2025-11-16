// Function: sub_8C6110
// Address: 0x8c6110
//
_DWORD *sub_8C6110()
{
  _DWORD *result; // rax
  __int64 (__fastcall *v1)(__int64, char); // r12
  __int64 v2; // rbx
  int v3; // edi
  _DWORD *v4; // [rsp-58h] [rbp-58h]
  int i; // [rsp-3Ch] [rbp-3Ch]

  result = qword_4D03FD0;
  if ( *qword_4D03FD0 )
  {
    v1 = sub_8C3600;
    for ( i = 2; ; i = 1 )
    {
      result = sub_759B50(
                 0,
                 0,
                 (__int64 (__fastcall *)(_QWORD, _QWORD))v1,
                 (__int64 (__fastcall *)(_QWORD, _QWORD))v1,
                 (__int64 (__fastcall *)(_QWORD, _QWORD))sub_8C31E0,
                 0);
      if ( dword_4F073A8 > 1 )
      {
        v2 = 0;
        v3 = 2;
        do
        {
          result = qword_4F073B0;
          if ( *((_QWORD *)qword_4F073B0 + v2 + 2) )
          {
            result = (_DWORD *)*((_QWORD *)qword_4F072B0 + v2 + 2);
            if ( (*(_BYTE *)(result - 2) & 2) == 0 )
            {
              if ( *((_BYTE *)result + 28) )
              {
                sub_75AFC0(
                  v3,
                  0,
                  0,
                  (__int64 (__fastcall *)(_QWORD, _QWORD))v1,
                  (__int64 (__fastcall *)(_QWORD, _QWORD))v1,
                  (__int64 (__fastcall *)(_QWORD, _QWORD))sub_8C31E0,
                  0);
                result = v4;
              }
            }
          }
          v3 = ++v2 + 2;
        }
        while ( dword_4F073A8 >= (int)v2 + 2 );
      }
      v1 = 0;
      if ( i == 1 )
        break;
    }
  }
  return result;
}
