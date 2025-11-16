// Function: sub_16E2D20
// Address: 0x16e2d20
//
void __fastcall sub_16E2D20(__int64 a1, __int64 a2, const char *a3, char a4)
{
  _BYTE *v6; // rax
  size_t v7; // rax
  void *v8; // rdi
  size_t v9; // r14
  size_t v10; // rdx
  const char *v11; // rsi
  void *v12; // rdi
  size_t v13; // r12
  const char *v14; // rsi
  const char *v15; // rsi

  switch ( a4 )
  {
    case 2:
      sub_16E2CE0((__int64)a3, a2);
      return;
    case 3:
      if ( a3 )
      {
        v7 = strlen(a3);
        v8 = *(void **)(a2 + 24);
        v9 = v7;
        if ( v7 > *(_QWORD *)(a2 + 16) - (_QWORD)v8 )
        {
          v10 = v7;
          v11 = a3;
          goto LABEL_13;
        }
        if ( v7 )
        {
          memcpy(v8, a3, v7);
          *(_QWORD *)(a2 + 24) += v9;
        }
      }
      return;
    case 4:
      v10 = *((_QWORD *)a3 + 1);
      goto LABEL_12;
    case 5:
      v12 = *(void **)(a2 + 24);
      v11 = *(const char **)a3;
      v13 = *((_QWORD *)a3 + 1);
      v10 = v13;
      if ( v13 > *(_QWORD *)(a2 + 16) - (_QWORD)v12 )
        goto LABEL_13;
      if ( v13 )
      {
        memcpy(v12, v11, v13);
        *(_QWORD *)(a2 + 24) += v13;
      }
      return;
    case 6:
      v10 = *((unsigned int *)a3 + 2);
LABEL_12:
      v11 = *(const char **)a3;
LABEL_13:
      sub_16E7EE0(a2, v11, v10);
      return;
    case 7:
      sub_16E8650(a2, a3);
      return;
    case 8:
      v6 = *(_BYTE **)(a2 + 24);
      if ( (unsigned __int64)v6 >= *(_QWORD *)(a2 + 16) )
      {
        sub_16E7DE0(a2, (unsigned __int8)a3);
      }
      else
      {
        *(_QWORD *)(a2 + 24) = v6 + 1;
        *v6 = (_BYTE)a3;
      }
      return;
    case 9:
      v14 = (const char *)(unsigned int)a3;
      goto LABEL_20;
    case 10:
      v15 = (const char *)(int)a3;
      goto LABEL_22;
    case 11:
      v14 = *(const char **)a3;
LABEL_20:
      sub_16E7A90(a2, v14);
      break;
    case 12:
      v15 = *(const char **)a3;
LABEL_22:
      sub_16E7AB0(a2, v15);
      break;
    case 13:
      sub_16E7AD0(a2);
      break;
    case 14:
      sub_16E7AF0(a2, *(_QWORD *)a3, a3);
      break;
    case 15:
      sub_16E7B10(a2, *(_QWORD *)a3);
      break;
    default:
      return;
  }
}
