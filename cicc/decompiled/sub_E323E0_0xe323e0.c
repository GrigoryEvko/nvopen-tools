// Function: sub_E323E0
// Address: 0xe323e0
//
void __fastcall sub_E323E0(__int64 a1)
{
  unsigned __int64 v1; // r12
  unsigned __int64 v2; // rax
  char v3; // cl
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // r13
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rsi
  int v10; // [rsp+Ch] [rbp-34h] BYREF
  size_t v11; // [rsp+10h] [rbp-30h] BYREF
  _BYTE *v12; // [rsp+18h] [rbp-28h]

  if ( *(_BYTE *)(a1 + 49) || (v1 = *(_QWORD *)(a1 + 8), v1 >= *(_QWORD *)a1) )
  {
    *(_BYTE *)(a1 + 49) = 1;
    return;
  }
  *(_QWORD *)(a1 + 8) = v1 + 1;
  v2 = *(_QWORD *)(a1 + 40);
  if ( v2 >= *(_QWORD *)(a1 + 24) )
  {
    *(_BYTE *)(a1 + 49) = 1;
    if ( !(unsigned __int8)sub_E313F0(0, &v10) )
      goto LABEL_24;
    goto LABEL_6;
  }
  *(_QWORD *)(a1 + 40) = v2 + 1;
  if ( (unsigned __int8)sub_E313F0(*(_BYTE *)(*(_QWORD *)(a1 + 32) + v2), &v10) )
  {
LABEL_6:
    if ( v10 > 13 )
    {
      if ( v10 == 17 )
      {
        sub_E31570(a1, 95);
        goto LABEL_25;
      }
    }
    else
    {
      if ( v10 > 1 )
      {
        if ( !*(_BYTE *)(a1 + 49) )
        {
          v8 = *(_QWORD *)(a1 + 40);
          if ( v8 < *(_QWORD *)(a1 + 24) && *(_BYTE *)(*(_QWORD *)(a1 + 32) + v8) == 110 )
          {
            *(_QWORD *)(a1 + 40) = v8 + 1;
            sub_E31570(a1, 45);
          }
        }
        v11 = 0;
        v12 = 0;
        v9 = sub_E322C0(a1, &v11);
        if ( v11 > 0x10 )
        {
          sub_E31C60(a1, 2u, "0x");
          sub_E31C60(a1, v11, v12);
        }
        else
        {
          sub_E31D10(a1, v9);
        }
        goto LABEL_25;
      }
      if ( v10 )
      {
        if ( v10 == 1 )
        {
          v11 = 0;
          v12 = 0;
          v4 = sub_E322C0(a1, &v11);
          if ( !*(_BYTE *)(a1 + 49) && v11 <= 6 )
          {
            sub_E31C60(a1, 1u, "'");
            if ( v4 > 0x27 )
            {
              if ( v4 == 92 )
              {
                sub_E31C60(a1, 2u, "\\\\");
LABEL_30:
                sub_E31570(a1, 39);
                goto LABEL_25;
              }
            }
            else if ( v4 > 8 )
            {
              switch ( v4 )
              {
                case 9uLL:
                  sub_E31C60(a1, 2u, "\\t");
                  break;
                case 0xAuLL:
                  sub_E31C60(a1, 2u, "\\n");
                  break;
                case 0xDuLL:
                  sub_E31C60(a1, 2u, "\\r");
                  break;
                case 0x22uLL:
                  sub_E31C60(a1, 1u, "\"");
                  break;
                case 0x27uLL:
                  sub_E31C60(a1, 2u, "\\'");
                  break;
                default:
                  goto LABEL_28;
              }
              goto LABEL_30;
            }
LABEL_28:
            if ( v4 - 32 > 0x5E )
            {
              sub_E31C60(a1, 3u, "\\u{");
              sub_E31C60(a1, v11, v12);
              sub_E31570(a1, 125);
            }
            else
            {
              sub_E31570(a1, v4);
            }
            goto LABEL_30;
          }
        }
      }
      else
      {
        v11 = 0;
        v12 = 0;
        sub_E322C0(a1, &v11);
        if ( v11 == 1 )
        {
          if ( *v12 == 48 )
          {
            sub_E31C60(a1, 5u, "false");
            goto LABEL_25;
          }
          if ( *v12 == 49 )
          {
            sub_E31C60(a1, 4u, "true");
            goto LABEL_25;
          }
        }
      }
    }
    goto LABEL_24;
  }
  if ( v3 == 66 )
  {
    if ( *(_BYTE *)(a1 + 49)
      || (v5 = *(_QWORD *)(a1 + 40), v5 >= *(_QWORD *)(a1 + 24))
      || *(_BYTE *)(*(_QWORD *)(a1 + 32) + v5) != 95 )
    {
      v7 = sub_E31BC0(a1);
      if ( *(_BYTE *)(a1 + 49) )
        goto LABEL_24;
      v6 = *(_QWORD *)(a1 + 40);
    }
    else
    {
      v6 = v5 + 1;
      v7 = 0;
      *(_QWORD *)(a1 + 40) = v6;
    }
    if ( v7 < v6 )
    {
      if ( *(_BYTE *)(a1 + 48) )
      {
        *(_QWORD *)(a1 + 40) = v7;
        sub_E323E0(a1);
        *(_QWORD *)(a1 + 40) = v6;
      }
      goto LABEL_25;
    }
  }
LABEL_24:
  *(_BYTE *)(a1 + 49) = 1;
LABEL_25:
  *(_QWORD *)(a1 + 8) = v1;
}
