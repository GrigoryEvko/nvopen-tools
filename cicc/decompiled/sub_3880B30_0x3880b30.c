// Function: sub_3880B30
// Address: 0x3880b30
//
void __fastcall sub_3880B30(unsigned __int64 *a1)
{
  unsigned __int64 v1; // rdx
  unsigned __int64 v3; // r14
  _BYTE *v4; // r12
  _BYTE *v5; // rbx
  unsigned __int64 v6; // rsi
  unsigned __int8 v7; // r13
  char v8; // di
  char v9; // al
  unsigned __int8 v10; // [rsp-51h] [rbp-51h]

  v1 = a1[1];
  if ( v1 )
  {
    v3 = *a1;
    v4 = (_BYTE *)(*a1 + v1);
    if ( (_BYTE *)*a1 != v4 )
    {
      v5 = (_BYTE *)*a1;
      v6 = *a1;
      while ( 1 )
      {
        while ( 1 )
        {
          ++v6;
          if ( *v5 == 92 )
            break;
          *(_BYTE *)(v6 - 1) = *v5++;
LABEL_5:
          if ( v5 == v4 )
            goto LABEL_21;
        }
        if ( v5 < v4 - 1 && v5[1] == 92 )
        {
          *(_BYTE *)(v6 - 1) = 92;
          v5 += 2;
          goto LABEL_5;
        }
        if ( v5 >= v4 - 2 || (v7 = v5[1], !isxdigit(v7)) || (v10 = v5[2], !isxdigit(v10)) )
        {
          *(_BYTE *)(v6 - 1) = 92;
          ++v5;
          goto LABEL_5;
        }
        if ( (unsigned __int8)(v7 - 48) <= 9u )
        {
          v8 = 16 * (v7 - 48);
        }
        else if ( (unsigned __int8)(v7 - 97) <= 5u )
        {
          v8 = 16 * v7 - 112;
        }
        else
        {
          v8 = -16;
          if ( (unsigned __int8)(v7 - 65) < 6u )
            v8 = 16 * v7 - 112;
        }
        v9 = v10 - 48;
        if ( (unsigned __int8)(v10 - 48) > 9u )
        {
          if ( (unsigned __int8)(v10 - 97) <= 5u )
          {
            v9 = v10 - 87;
          }
          else
          {
            v9 = v10 - 55;
            if ( (unsigned __int8)(v10 - 65) >= 6u )
              v9 = -1;
          }
        }
        v5 += 3;
        *(_BYTE *)(v6 - 1) = v8 + v9;
        if ( v5 == v4 )
          goto LABEL_21;
      }
    }
    v6 = *a1;
LABEL_21:
    sub_22410F0(a1, v6 - v3, 0);
  }
}
