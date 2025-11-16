// Function: sub_5C9690
// Address: 0x5c9690
//
__int64 __fastcall sub_5C9690(unsigned __int64 a1, _BYTE *a2)
{
  unsigned __int64 v2; // r13
  __int64 result; // rax
  _BYTE *v4; // rax
  int v5; // ebx
  __int64 v6; // rdx
  __int64 v7; // rax
  char v8; // dl
  __int64 v9[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = a1;
  v9[0] = (__int64)a2;
  if ( *a2 == 40 && a2[1] == 124 )
  {
    v4 = a2 + 1;
    v9[0] = (__int64)(a2 + 1);
    goto LABEL_5;
  }
  result = sub_5C9390(a1, v9);
  if ( !(_DWORD)result )
  {
    v4 = (_BYTE *)v9[0];
LABEL_5:
    v5 = 0;
    while ( 1 )
    {
      if ( *v4 != 124 )
        return 0;
      v6 = (__int64)(v4 + 1);
      v7 = (__int64)(v4 + 2);
      v9[0] = v6;
      v8 = *(_BYTE *)(v7 - 1);
      if ( v8 == 77 )
      {
        v9[0] = v7;
        v5 = 0;
        v2 = qword_4F07788;
        sub_5C9390(qword_4F07788, v9);
        goto LABEL_10;
      }
      if ( v8 > 77 )
      {
        if ( v8 == 83 )
        {
          v9[0] = v7;
          v2 = unk_4F07778;
          if ( (unsigned int)sub_5C9390(unk_4F07778, v9) )
            return 1;
          v5 = 1;
          goto LABEL_10;
        }
      }
      else if ( v8 == 67 )
      {
        v2 = unk_4F077A0;
        v5 = unk_4F077B4;
      }
      else if ( v8 == 71 )
      {
        v2 = unk_4F077A8;
        v5 = unk_4F077B8;
      }
      v9[0] = v7;
      if ( (unsigned int)sub_5C9390(v2, v9) && v5 )
        return 1;
LABEL_10:
      v4 = (_BYTE *)v9[0];
    }
  }
  return result;
}
