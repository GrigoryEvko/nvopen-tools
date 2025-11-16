// Function: sub_1296350
// Address: 0x1296350
//
void __fastcall sub_1296350(__int64 *a1, unsigned __int64 a2)
{
  bool v2; // zf
  char v3; // bl
  __int64 v4; // rdx
  __int64 *v5; // rcx
  __int64 v6; // r8
  _QWORD *v7; // rax
  __int64 v8; // [rsp+8h] [rbp-48h] BYREF
  _BYTE v9[64]; // [rsp+10h] [rbp-40h] BYREF

  v2 = a1[7] == 0;
  v8 = *(_QWORD *)a2;
  if ( v2 )
  {
    v7 = (_QWORD *)sub_12A4D50(a1, byte_3F871B3, 0, 0);
    sub_1290AF0(a1, v7, 0);
  }
  v3 = *(_BYTE *)(a2 + 40);
  if ( v3 == 8 )
  {
    if ( !(_DWORD)v8 && !WORD2(v8) )
      v8 = *(_QWORD *)(*(_QWORD *)(a1[55] + 80) + 8LL);
    sub_1290930((__int64)a1, (unsigned int *)&v8);
    sub_127C770(&v8);
    sub_1291AE0((__int64)a1, a2);
  }
  else
  {
    sub_1290930((__int64)a1, (unsigned int *)&v8);
    sub_127C770(&v8);
    switch ( v3 )
    {
      case 0:
      case 25:
        sub_127FF60((__int64)v9, (__int64)a1, *(__int64 **)(a2 + 48), 0, 0, 0);
        break;
      case 1:
        sub_1296FA0(a1, a2);
        break;
      case 2:
        sub_1296F00(a1, a2);
        break;
      case 5:
        sub_1297100(a1, a2);
        break;
      case 6:
        sub_1291990((__int64)a1, a2);
        break;
      case 7:
        sub_1291130(a1);
        break;
      case 11:
        sub_1296570((unsigned int)v9, (_DWORD)a1, a2, 0, 0, 0, 0);
        break;
      case 12:
        sub_1296AD0(a1, a2);
        break;
      case 13:
        sub_1296CC0(a1, a2);
        break;
      case 15:
        sub_12955E0((__int64)a1, a2);
        break;
      case 16:
        sub_1295900((__int64)a1, (_QWORD *)a2);
        break;
      case 17:
        sub_1290F50(a1, a2);
        break;
      case 18:
        sub_1292420(a1, a2, v4, v5, v6);
        break;
      case 20:
        sub_1291D60((__int64)a1, a2);
        break;
      case 24:
        return;
      default:
        sub_127B550("unsupported statement type", (_DWORD *)a2, 1);
    }
  }
}
