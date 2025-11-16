// Function: sub_16F1A20
// Address: 0x16f1a20
//
_BYTE *__fastcall sub_16F1A20(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  unsigned __int8 *v3; // r15
  _BYTE *v5; // rax
  unsigned __int8 *v6; // r13
  _BYTE *v7; // rax
  unsigned __int64 v8; // rdx
  unsigned __int8 v9; // bl
  unsigned __int8 *v10; // rax
  unsigned __int64 v11; // rdx
  _BYTE *result; // rax
  _BYTE *v13; // rax
  _QWORD v14[8]; // [rsp+0h] [rbp-40h] BYREF

  v3 = a2;
  v5 = *(_BYTE **)(a1 + 24);
  if ( (unsigned __int64)v5 >= *(_QWORD *)(a1 + 16) )
  {
    sub_16E7DE0(a1, 34);
  }
  else
  {
    *(_QWORD *)(a1 + 24) = v5 + 1;
    *v5 = 34;
  }
  v6 = &a2[a3];
  if ( a2 != v6 )
  {
    do
    {
      while ( 1 )
      {
        v9 = *v3;
        if ( *v3 == 34 || v9 == 92 )
        {
          v13 = *(_BYTE **)(a1 + 24);
          if ( (unsigned __int64)v13 >= *(_QWORD *)(a1 + 16) )
          {
            sub_16E7DE0(a1, 92);
          }
          else
          {
            *(_QWORD *)(a1 + 24) = v13 + 1;
            *v13 = 92;
          }
        }
        v10 = *(unsigned __int8 **)(a1 + 24);
        v11 = *(_QWORD *)(a1 + 16);
        if ( v9 > 0x1Fu )
          break;
        if ( v11 <= (unsigned __int64)v10 )
        {
          sub_16E7DE0(a1, 92);
          v7 = *(_BYTE **)(a1 + 24);
          v8 = *(_QWORD *)(a1 + 16);
          if ( v9 != 10 )
          {
LABEL_7:
            if ( v9 == 13 )
            {
              if ( v8 <= (unsigned __int64)v7 )
              {
                sub_16E7DE0(a1, 114);
              }
              else
              {
                *(_QWORD *)(a1 + 24) = v7 + 1;
                *v7 = 114;
              }
            }
            else if ( v9 == 9 )
            {
              if ( v8 <= (unsigned __int64)v7 )
              {
                sub_16E7DE0(a1, 116);
              }
              else
              {
                *(_QWORD *)(a1 + 24) = v7 + 1;
                *v7 = 116;
              }
            }
            else
            {
              if ( v8 <= (unsigned __int64)v7 )
              {
                sub_16E7DE0(a1, 117);
              }
              else
              {
                *(_QWORD *)(a1 + 24) = v7 + 1;
                *v7 = 117;
              }
              v14[0] = 4;
              v14[1] = 1;
              sub_16F4F70(a1, v9, 1, v14);
            }
            goto LABEL_12;
          }
        }
        else
        {
          *(_QWORD *)(a1 + 24) = v10 + 1;
          *v10 = 92;
          v7 = *(_BYTE **)(a1 + 24);
          v8 = *(_QWORD *)(a1 + 16);
          if ( v9 != 10 )
            goto LABEL_7;
        }
        if ( v8 <= (unsigned __int64)v7 )
        {
          sub_16E7DE0(a1, 110);
        }
        else
        {
          *(_QWORD *)(a1 + 24) = v7 + 1;
          *v7 = 110;
        }
LABEL_12:
        if ( v6 == ++v3 )
          goto LABEL_18;
      }
      if ( v11 <= (unsigned __int64)v10 )
      {
        sub_16E7DE0(a1, v9);
        goto LABEL_12;
      }
      ++v3;
      *(_QWORD *)(a1 + 24) = v10 + 1;
      *v10 = v9;
    }
    while ( v6 != v3 );
  }
LABEL_18:
  result = *(_BYTE **)(a1 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(a1 + 16) )
    return (_BYTE *)sub_16E7DE0(a1, 34);
  *(_QWORD *)(a1 + 24) = result + 1;
  *result = 34;
  return result;
}
