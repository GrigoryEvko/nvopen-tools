// Function: sub_CB6190
// Address: 0xcb6190
//
void *__fastcall sub_CB6190(__int64 a1, unsigned __int8 *a2, size_t a3)
{
  _BYTE *v5; // rdi
  void *result; // rax

  v5 = *(_BYTE **)(a1 + 32);
  switch ( a3 )
  {
    case 0uLL:
      goto LABEL_6;
    case 1uLL:
      goto LABEL_5;
    case 2uLL:
      goto LABEL_4;
    case 3uLL:
      goto LABEL_3;
    case 4uLL:
      v5[3] = a2[3];
      v5 = *(_BYTE **)(a1 + 32);
LABEL_3:
      v5[2] = a2[2];
      v5 = *(_BYTE **)(a1 + 32);
LABEL_4:
      v5[1] = a2[1];
      v5 = *(_BYTE **)(a1 + 32);
LABEL_5:
      result = (void *)*a2;
      *v5 = (_BYTE)result;
LABEL_6:
      *(_QWORD *)(a1 + 32) += a3;
      break;
    default:
      result = memcpy(v5, a2, a3);
      *(_QWORD *)(a1 + 32) += a3;
      break;
  }
  return result;
}
