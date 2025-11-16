// Function: sub_646240
// Address: 0x646240
//
__int64 __fastcall sub_646240(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  int v4; // edx
  char v5; // dl
  __int64 v6; // r12
  unsigned int v7; // r14d
  unsigned __int8 v8; // r13
  __int64 v9; // rdi

  result = *(_DWORD *)(a1 + 16) & 0x20008;
  if ( (_DWORD)result == 8 )
  {
    v4 = *(unsigned __int8 *)(a1 + 56);
    result = (unsigned int)(v4 - 2);
    v5 = v4 & 0xFD;
    if ( (result & 0xFD) == 0 || v5 == 1 )
    {
      v6 = a2 + 88;
      result = *(_QWORD *)(a2 + 8) & 2LL;
      if ( !dword_4F04C34 || (*(_BYTE *)(a1 + 16) & 5) == 5 )
      {
        v7 = 0;
        v8 = 3;
        if ( *(_BYTE *)(a2 + 269) == 2 )
        {
          if ( dword_4D04964 )
          {
            v7 = 984;
            v8 = byte_4F07472[0];
          }
          else
          {
            v8 = 5;
            v7 = 984;
          }
        }
      }
      else
      {
        if ( !dword_4F077BC )
        {
          if ( v5 == 1 )
          {
            if ( !result )
            {
              v7 = 732;
              v8 = 8;
LABEL_9:
              result = sub_684AC0(v8, v7);
              if ( v8 == 8 )
              {
                *(_BYTE *)(a1 + 17) |= 0x20u;
                *(_QWORD *)(a1 + 24) = 0;
              }
              return result;
            }
            v8 = 8;
            v7 = 732;
LABEL_20:
            result = sub_8D2310(*(_QWORD *)(a2 + 288));
            if ( (_DWORD)result )
            {
              if ( (*(_BYTE *)(a2 + 122) & 4) == 0 )
              {
                result = sub_6461D0(a1, *(_QWORD *)(a2 + 288), 1);
                if ( (_DWORD)result )
                {
                  v9 = 5;
                  if ( dword_4D04964 )
                    v9 = byte_4F07472[0];
                  result = sub_684AA0(v9, 2620, v6);
                }
              }
            }
LABEL_8:
            if ( !v7 )
              return result;
            goto LABEL_9;
          }
          v8 = 8;
          v7 = 733;
LABEL_7:
          if ( !result )
            goto LABEL_8;
          goto LABEL_20;
        }
        if ( qword_4F077A8 <= 0x9C3Fu )
        {
          v8 = 5;
          v7 = (v5 != 1) + 732;
          goto LABEL_19;
        }
        v8 = 8;
        v7 = 733;
        if ( v5 == 1 )
        {
          v7 = 732;
          goto LABEL_19;
        }
      }
      if ( !dword_4F077BC )
        goto LABEL_7;
LABEL_19:
      if ( !result )
      {
        result = sub_6440B0(0x1Cu, a2);
        v6 = result + 56;
        if ( !result )
          goto LABEL_8;
      }
      goto LABEL_20;
    }
  }
  return result;
}
