// Function: sub_39E0070
// Address: 0x39e0070
//
_WORD *__fastcall sub_39E0070(unsigned __int8 *a1, int a2, __int64 a3)
{
  unsigned __int8 *v3; // r14
  _BYTE *v5; // rax
  _WORD *result; // rax
  __int64 v7; // r12
  unsigned __int8 v8; // bl
  unsigned __int64 v9; // rdx
  char *v10; // rax
  char v11; // si
  char *v12; // rax
  char v13; // si
  __int64 v14; // rdi
  unsigned __int8 *v15; // rax

  v3 = a1;
  v5 = *(_BYTE **)(a3 + 24);
  if ( (unsigned __int64)v5 >= *(_QWORD *)(a3 + 16) )
  {
    sub_16E7DE0(a3, 34);
  }
  else
  {
    *(_QWORD *)(a3 + 24) = v5 + 1;
    *v5 = 34;
  }
  result = *(_WORD **)(a3 + 24);
  if ( a2 )
  {
    v7 = (__int64)&a1[a2 - 1 + 1];
    while ( 2 )
    {
      v8 = *v3;
      v9 = *(_QWORD *)(a3 + 16);
      if ( *v3 == 34 )
        break;
LABEL_6:
      if ( v8 != 92 )
      {
        if ( (unsigned __int8)(v8 - 32) > 0x5Eu )
        {
          switch ( v8 )
          {
            case 8u:
              if ( v9 - (unsigned __int64)result <= 1 )
              {
                sub_16E7EE0(a3, "\\b", 2u);
                result = *(_WORD **)(a3 + 24);
              }
              else
              {
                *result = 25180;
                result = (_WORD *)(*(_QWORD *)(a3 + 24) + 2LL);
                *(_QWORD *)(a3 + 24) = result;
              }
              goto LABEL_29;
            case 9u:
              if ( v9 - (unsigned __int64)result <= 1 )
              {
                sub_16E7EE0(a3, "\\t", 2u);
                result = *(_WORD **)(a3 + 24);
              }
              else
              {
                *result = 29788;
                result = (_WORD *)(*(_QWORD *)(a3 + 24) + 2LL);
                *(_QWORD *)(a3 + 24) = result;
              }
              goto LABEL_29;
            case 0xAu:
              if ( v9 - (unsigned __int64)result <= 1 )
              {
                sub_16E7EE0(a3, "\\n", 2u);
                result = *(_WORD **)(a3 + 24);
              }
              else
              {
                *result = 28252;
                result = (_WORD *)(*(_QWORD *)(a3 + 24) + 2LL);
                *(_QWORD *)(a3 + 24) = result;
              }
              goto LABEL_29;
            case 0xCu:
              if ( v9 - (unsigned __int64)result <= 1 )
              {
                sub_16E7EE0(a3, "\\f", 2u);
                result = *(_WORD **)(a3 + 24);
              }
              else
              {
                *result = 26204;
                result = (_WORD *)(*(_QWORD *)(a3 + 24) + 2LL);
                *(_QWORD *)(a3 + 24) = result;
              }
              goto LABEL_29;
            case 0xDu:
              if ( v9 - (unsigned __int64)result <= 1 )
              {
                sub_16E7EE0(a3, "\\r", 2u);
                result = *(_WORD **)(a3 + 24);
                goto LABEL_29;
              }
              ++v3;
              *result = 29276;
              result = (_WORD *)(*(_QWORD *)(a3 + 24) + 2LL);
              *(_QWORD *)(a3 + 24) = result;
              if ( (unsigned __int8 *)v7 == v3 )
                goto LABEL_11;
              continue;
            default:
              if ( v9 <= (unsigned __int64)result )
              {
                sub_16E7DE0(a3, 92);
              }
              else
              {
                *(_QWORD *)(a3 + 24) = (char *)result + 1;
                *(_BYTE *)result = 92;
              }
              v10 = *(char **)(a3 + 24);
              v11 = (v8 >> 6) + 48;
              if ( (unsigned __int64)v10 >= *(_QWORD *)(a3 + 16) )
              {
                sub_16E7DE0(a3, v11);
              }
              else
              {
                *(_QWORD *)(a3 + 24) = v10 + 1;
                *v10 = v11;
              }
              v12 = *(char **)(a3 + 24);
              v13 = ((v8 >> 3) & 7) + 48;
              if ( (unsigned __int64)v12 >= *(_QWORD *)(a3 + 16) )
              {
                sub_16E7DE0(a3, v13);
              }
              else
              {
                *(_QWORD *)(a3 + 24) = v12 + 1;
                *v12 = v13;
              }
              result = *(_WORD **)(a3 + 24);
              v8 = (v8 & 7) + 48;
              if ( (unsigned __int64)result < *(_QWORD *)(a3 + 16) )
                goto LABEL_28;
              goto LABEL_37;
          }
        }
        if ( v9 > (unsigned __int64)result )
        {
LABEL_28:
          *(_QWORD *)(a3 + 24) = (char *)result + 1;
          *(_BYTE *)result = v8;
          result = *(_WORD **)(a3 + 24);
        }
        else
        {
LABEL_37:
          sub_16E7DE0(a3, v8);
          result = *(_WORD **)(a3 + 24);
        }
        goto LABEL_29;
      }
      break;
    }
    while ( 1 )
    {
      if ( v9 <= (unsigned __int64)result )
      {
        v14 = sub_16E7DE0(a3, 92);
      }
      else
      {
        v14 = a3;
        *(_QWORD *)(a3 + 24) = (char *)result + 1;
        *(_BYTE *)result = 92;
      }
      v15 = *(unsigned __int8 **)(v14 + 24);
      if ( (unsigned __int64)v15 >= *(_QWORD *)(v14 + 16) )
      {
        sub_16E7DE0(v14, v8);
      }
      else
      {
        *(_QWORD *)(v14 + 24) = v15 + 1;
        *v15 = v8;
      }
      result = *(_WORD **)(a3 + 24);
LABEL_29:
      if ( (unsigned __int8 *)v7 == ++v3 )
        break;
      v8 = *v3;
      v9 = *(_QWORD *)(a3 + 16);
      if ( *v3 != 34 )
        goto LABEL_6;
    }
  }
LABEL_11:
  if ( (unsigned __int64)result >= *(_QWORD *)(a3 + 16) )
    return (_WORD *)sub_16E7DE0(a3, 34);
  *(_QWORD *)(a3 + 24) = (char *)result + 1;
  *(_BYTE *)result = 34;
  return result;
}
