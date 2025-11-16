// Function: sub_CA1350
// Address: 0xca1350
//
void __fastcall sub_CA1350(__int64 a1, __int64 a2, const char *a3, size_t a4, char a5)
{
  _BYTE *v7; // rax
  void *v8; // rdi
  __int64 v9; // rax
  size_t v10; // rdx
  const char *v11; // rsi
  const char *v12; // rsi
  const char *v13; // rsi
  size_t v14; // rax
  void *v15; // rdi
  size_t v16; // r14

  switch ( a5 )
  {
    case 2:
      sub_CA0E80((__int64)a3, a2);
      return;
    case 3:
      if ( a3 )
      {
        v14 = strlen(a3);
        v15 = *(void **)(a2 + 32);
        v16 = v14;
        if ( v14 > *(_QWORD *)(a2 + 24) - (_QWORD)v15 )
        {
          v10 = v14;
          v11 = a3;
          goto LABEL_24;
        }
        if ( v14 )
        {
          memcpy(v15, a3, v14);
          *(_QWORD *)(a2 + 32) += v16;
        }
      }
      return;
    case 4:
      v10 = *((_QWORD *)a3 + 1);
      v11 = *(const char **)a3;
      goto LABEL_24;
    case 5:
    case 6:
      v8 = *(void **)(a2 + 32);
      v9 = *(_QWORD *)(a2 + 24);
      v10 = a4;
      v11 = a3;
      if ( a4 > v9 - (__int64)v8 )
      {
LABEL_24:
        sub_CB6200(a2, v11, v10);
      }
      else if ( a4 )
      {
        memcpy(v8, a3, a4);
        *(_QWORD *)(a2 + 32) += a4;
      }
      return;
    case 7:
      sub_CB6840(a2, a3);
      return;
    case 8:
      v7 = *(_BYTE **)(a2 + 32);
      if ( (unsigned __int64)v7 >= *(_QWORD *)(a2 + 24) )
      {
        sub_CB5D20(a2, (unsigned __int8)a3);
      }
      else
      {
        *(_QWORD *)(a2 + 32) = v7 + 1;
        *v7 = (_BYTE)a3;
      }
      return;
    case 9:
      v12 = (const char *)(unsigned int)a3;
      goto LABEL_10;
    case 10:
      v13 = (const char *)(int)a3;
      goto LABEL_12;
    case 11:
      v12 = *(const char **)a3;
LABEL_10:
      sub_CB59D0(a2, v12);
      break;
    case 12:
      v13 = *(const char **)a3;
LABEL_12:
      sub_CB59F0(a2, v13);
      break;
    case 13:
      sub_CB5A10(a2, *(_QWORD *)a3);
      break;
    case 14:
      sub_CB5A30(a2, *(_QWORD *)a3);
      break;
    case 15:
      sub_CB5A50(a2, *(_QWORD *)a3);
      break;
    default:
      return;
  }
}
