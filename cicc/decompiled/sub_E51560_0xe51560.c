// Function: sub_E51560
// Address: 0xe51560
//
_WORD *__fastcall sub_E51560(__int64 a1, char *a2, __int64 a3, __int64 a4)
{
  char *v6; // rbx
  _BYTE *v7; // rax
  char *v8; // r13
  _WORD *result; // rax
  char v10; // si
  unsigned __int64 v11; // rdx
  unsigned __int8 v12; // r12
  _BYTE *v13; // rax
  char *v14; // rax
  char v15; // si
  char *v16; // rax
  char v17; // si
  unsigned __int8 *v18; // rax
  _BYTE *v19; // rax
  __int64 v20; // rdi
  unsigned __int8 *v21; // rax

  v6 = a2;
  v7 = *(_BYTE **)(a4 + 32);
  if ( (unsigned __int64)v7 >= *(_QWORD *)(a4 + 24) )
  {
    sub_CB5D20(a4, 34);
  }
  else
  {
    *(_QWORD *)(a4 + 32) = v7 + 1;
    *v7 = 34;
  }
  v8 = &a2[a3];
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 312) + 21LL) )
  {
    result = *(_WORD **)(a4 + 32);
    if ( a2 != v8 )
    {
      while ( 1 )
      {
        v10 = *v6;
        v11 = *(_QWORD *)(a4 + 24);
        if ( *v6 != 34 )
          break;
        if ( v11 - (unsigned __int64)result <= 1 )
        {
          sub_CB6200(a4, (unsigned __int8 *)"\"\"", 2u);
          result = *(_WORD **)(a4 + 32);
LABEL_9:
          if ( v8 == ++v6 )
            goto LABEL_13;
        }
        else
        {
          ++v6;
          *result = 8738;
          result = (_WORD *)(*(_QWORD *)(a4 + 32) + 2LL);
          *(_QWORD *)(a4 + 32) = result;
          if ( v8 == v6 )
            goto LABEL_13;
        }
      }
      if ( v11 <= (unsigned __int64)result )
      {
        sub_CB5D20(a4, v10);
      }
      else
      {
        *(_QWORD *)(a4 + 32) = (char *)result + 1;
        *(_BYTE *)result = v10;
      }
      result = *(_WORD **)(a4 + 32);
      goto LABEL_9;
    }
LABEL_13:
    if ( (unsigned __int64)result < *(_QWORD *)(a4 + 24) )
      goto LABEL_14;
    return (_WORD *)sub_CB5D20(a4, 34);
  }
  if ( a2 != v8 )
  {
    do
    {
      v12 = *v6;
      if ( *v6 != 34 && v12 != 92 )
      {
        if ( (unsigned __int8)(v12 - 32) <= 0x5Eu )
        {
LABEL_27:
          v18 = *(unsigned __int8 **)(a4 + 32);
          if ( (unsigned __int64)v18 >= *(_QWORD *)(a4 + 24) )
          {
            sub_CB5D20(a4, v12);
          }
          else
          {
            *(_QWORD *)(a4 + 32) = v18 + 1;
            *v18 = v12;
          }
        }
        else
        {
          switch ( v12 )
          {
            case 8u:
              sub_904010(a4, "\\b");
              break;
            case 9u:
              sub_904010(a4, "\\t");
              break;
            case 0xAu:
              sub_904010(a4, "\\n");
              break;
            case 0xCu:
              sub_904010(a4, "\\f");
              break;
            case 0xDu:
              sub_904010(a4, "\\r");
              break;
            default:
              v13 = *(_BYTE **)(a4 + 32);
              if ( (unsigned __int64)v13 >= *(_QWORD *)(a4 + 24) )
              {
                sub_CB5D20(a4, 92);
              }
              else
              {
                *(_QWORD *)(a4 + 32) = v13 + 1;
                *v13 = 92;
              }
              v14 = *(char **)(a4 + 32);
              v15 = (v12 >> 6) + 48;
              if ( (unsigned __int64)v14 >= *(_QWORD *)(a4 + 24) )
              {
                sub_CB5D20(a4, v15);
              }
              else
              {
                *(_QWORD *)(a4 + 32) = v14 + 1;
                *v14 = v15;
              }
              v16 = *(char **)(a4 + 32);
              v17 = ((v12 >> 3) & 7) + 48;
              if ( (unsigned __int64)v16 >= *(_QWORD *)(a4 + 24) )
              {
                sub_CB5D20(a4, v17);
              }
              else
              {
                *(_QWORD *)(a4 + 32) = v16 + 1;
                *v16 = v17;
              }
              v12 = (v12 & 7) + 48;
              goto LABEL_27;
          }
        }
        goto LABEL_29;
      }
      v19 = *(_BYTE **)(a4 + 32);
      if ( (unsigned __int64)v19 >= *(_QWORD *)(a4 + 24) )
      {
        v20 = sub_CB5D20(a4, 92);
        v21 = *(unsigned __int8 **)(v20 + 32);
        if ( (unsigned __int64)v21 >= *(_QWORD *)(v20 + 24) )
        {
LABEL_45:
          sub_CB5D20(v20, v12);
          goto LABEL_29;
        }
      }
      else
      {
        v20 = a4;
        *(_QWORD *)(a4 + 32) = v19 + 1;
        *v19 = 92;
        v21 = *(unsigned __int8 **)(a4 + 32);
        if ( (unsigned __int64)v21 >= *(_QWORD *)(a4 + 24) )
          goto LABEL_45;
      }
      *(_QWORD *)(v20 + 32) = v21 + 1;
      *v21 = v12;
LABEL_29:
      ++v6;
    }
    while ( v8 != v6 );
  }
  result = *(_WORD **)(a4 + 32);
  if ( (unsigned __int64)result < *(_QWORD *)(a4 + 24) )
  {
LABEL_14:
    *(_QWORD *)(a4 + 32) = (char *)result + 1;
    *(_BYTE *)result = 34;
    return result;
  }
  return (_WORD *)sub_CB5D20(a4, 34);
}
