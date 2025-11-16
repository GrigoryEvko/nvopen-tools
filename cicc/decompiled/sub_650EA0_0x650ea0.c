// Function: sub_650EA0
// Address: 0x650ea0
//
unsigned __int16 *__fastcall sub_650EA0(__int64 a1, __int64 a2)
{
  unsigned __int16 *result; // rax
  __int64 v3; // rax
  __int64 *v4; // r14
  __int64 v5; // rbx
  int v6; // r15d
  int v7; // edx
  int v8; // [rsp-3Ch] [rbp-3Ch]

  result = (unsigned __int16 *)dword_4F077BC;
  if ( dword_4F077BC )
  {
    result = word_4F06418;
    if ( word_4F06418[0] == 142 )
    {
      v3 = sub_5CC190(14);
      v4 = (__int64 *)v3;
      if ( v3 )
      {
        v5 = v3;
        v6 = 0;
        v7 = 0;
        do
        {
          if ( *(_BYTE *)(v5 + 9) == 2 || (*(_BYTE *)(v5 + 11) & 0x10) != 0 )
          {
            if ( (unsigned __int64)(qword_4F077A8 - 30100LL) > 0x12B )
            {
              if ( !v6 )
              {
                v8 = v7;
                sub_684B30(1359, v5 + 56);
                v7 = v8;
              }
              *(_BYTE *)(v5 + 8) = 0;
              v6 = 1;
            }
          }
          else
          {
            if ( !v7 )
              sub_6851C0(1359, v5 + 56);
            *(_BYTE *)(v5 + 8) = 0;
            v7 = 1;
          }
          v5 = *(_QWORD *)v5;
        }
        while ( v5 );
      }
      sub_5CF700(v4);
      sub_6447A0(a2);
      sub_5CEC90(v4, a1, 7);
      return (unsigned __int16 *)sub_6447E0(a2);
    }
  }
  return result;
}
