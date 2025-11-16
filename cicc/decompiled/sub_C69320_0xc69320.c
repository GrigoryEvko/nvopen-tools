// Function: sub_C69320
// Address: 0xc69320
//
_BYTE *__fastcall sub_C69320(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  unsigned __int8 *v4; // r13
  _BYTE *v5; // rax
  unsigned __int8 *v6; // r14
  _BYTE *v7; // rax
  unsigned __int64 v8; // rdx
  unsigned __int8 v9; // bl
  unsigned __int8 *v10; // rax
  unsigned __int64 v11; // rdx
  _BYTE *result; // rax
  _BYTE *v13; // rax

  v4 = a2;
  v5 = *(_BYTE **)(a1 + 32);
  if ( (unsigned __int64)v5 >= *(_QWORD *)(a1 + 24) )
  {
    sub_CB5D20(a1, 34);
  }
  else
  {
    *(_QWORD *)(a1 + 32) = v5 + 1;
    *v5 = 34;
  }
  v6 = &a2[a3];
  if ( a2 != v6 )
  {
    do
    {
      while ( 1 )
      {
        v9 = *v4;
        if ( *v4 == 34 || v9 == 92 )
        {
          v13 = *(_BYTE **)(a1 + 32);
          if ( (unsigned __int64)v13 >= *(_QWORD *)(a1 + 24) )
          {
            sub_CB5D20(a1, 92);
          }
          else
          {
            *(_QWORD *)(a1 + 32) = v13 + 1;
            *v13 = 92;
          }
        }
        v10 = *(unsigned __int8 **)(a1 + 32);
        v11 = *(_QWORD *)(a1 + 24);
        if ( v9 > 0x1Fu )
          break;
        if ( v11 <= (unsigned __int64)v10 )
        {
          sub_CB5D20(a1, 92);
          v7 = *(_BYTE **)(a1 + 32);
          v8 = *(_QWORD *)(a1 + 24);
          if ( v9 != 10 )
          {
LABEL_7:
            if ( v9 == 13 )
            {
              if ( v8 <= (unsigned __int64)v7 )
              {
                sub_CB5D20(a1, 114);
              }
              else
              {
                *(_QWORD *)(a1 + 32) = v7 + 1;
                *v7 = 114;
              }
            }
            else if ( v9 == 9 )
            {
              if ( v8 <= (unsigned __int64)v7 )
              {
                sub_CB5D20(a1, 116);
              }
              else
              {
                *(_QWORD *)(a1 + 32) = v7 + 1;
                *v7 = 116;
              }
            }
            else
            {
              if ( v8 <= (unsigned __int64)v7 )
              {
                sub_CB5D20(a1, 117);
              }
              else
              {
                *(_QWORD *)(a1 + 32) = v7 + 1;
                *v7 = 117;
              }
              sub_C7F500(a1, v9, 1, 4, 1);
            }
            goto LABEL_12;
          }
        }
        else
        {
          *(_QWORD *)(a1 + 32) = v10 + 1;
          *v10 = 92;
          v7 = *(_BYTE **)(a1 + 32);
          v8 = *(_QWORD *)(a1 + 24);
          if ( v9 != 10 )
            goto LABEL_7;
        }
        if ( v8 <= (unsigned __int64)v7 )
        {
          sub_CB5D20(a1, 110);
        }
        else
        {
          *(_QWORD *)(a1 + 32) = v7 + 1;
          *v7 = 110;
        }
LABEL_12:
        if ( v6 == ++v4 )
          goto LABEL_18;
      }
      if ( v11 <= (unsigned __int64)v10 )
      {
        sub_CB5D20(a1, v9);
        goto LABEL_12;
      }
      ++v4;
      *(_QWORD *)(a1 + 32) = v10 + 1;
      *v10 = v9;
    }
    while ( v6 != v4 );
  }
LABEL_18:
  result = *(_BYTE **)(a1 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(a1 + 24) )
    return (_BYTE *)sub_CB5D20(a1, 34);
  *(_QWORD *)(a1 + 32) = result + 1;
  *result = 34;
  return result;
}
