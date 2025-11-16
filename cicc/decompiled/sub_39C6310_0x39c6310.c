// Function: sub_39C6310
// Address: 0x39c6310
//
void __fastcall sub_39C6310(__int64 a1, __int64 a2, unsigned __int16 a3)
{
  unsigned __int16 v3; // r13
  unsigned int v4; // eax
  __int64 *v5; // rax
  size_t *v6; // rsi
  size_t v7; // rdx
  int *v8; // rsi
  __int64 v9; // rax
  __int64 *v10[6]; // [rsp+0h] [rbp-30h] BYREF

  v3 = *(_WORD *)(a2 + 4);
  switch ( *(_DWORD *)a2 )
  {
    case 1:
      sub_39C2ED0((int *)a1, 0x41u);
      sub_39C2ED0((int *)a1, v3);
      switch ( *(_WORD *)(a2 + 6) )
      {
        case 5:
        case 6:
        case 7:
        case 0xB:
        case 0xD:
        case 0xF:
          sub_39C2ED0((int *)a1, 0xDu);
          sub_39C2F50((int *)a1, *(_QWORD *)(a2 + 8));
          break;
        case 8:
        case 9:
        case 0xA:
        case 0xC:
        case 0xE:
        case 0x10:
        case 0x11:
        case 0x12:
        case 0x13:
        case 0x14:
        case 0x15:
        case 0x16:
        case 0x17:
        case 0x18:
        case 0x19:
          sub_39C2ED0((int *)a1, 0xCu);
          sub_39C2ED0((int *)a1, *(_QWORD *)(a2 + 8));
          break;
      }
      return;
    case 2:
      sub_39C2ED0((int *)a1, 0x41u);
      sub_39C2ED0((int *)a1, v3);
      sub_39C2ED0((int *)a1, 8u);
      v6 = *(size_t **)(a2 + 8);
      v7 = *v6;
      v8 = (int *)(v6 + 3);
      goto LABEL_12;
    case 6:
      sub_39C5FF0(a1, v3, a3, *(_QWORD *)(a2 + 8));
      return;
    case 7:
    case 8:
    case 9:
      sub_39C2ED0((int *)a1, 0x41u);
      sub_39C2ED0((int *)a1, v3);
      sub_39C2ED0((int *)a1, 9u);
      if ( *(_DWORD *)a2 == 7 )
      {
        v4 = sub_3982A70(*(__int64 ***)(a2 + 8), *(_QWORD *)(a1 + 152));
      }
      else
      {
        if ( *(_DWORD *)a2 != 8 )
        {
          sub_39C5B70(a1, (__int64 *)(a2 + 8));
          return;
        }
        v4 = sub_3982A00(*(__int64 ***)(a2 + 8), *(_QWORD *)(a1 + 152));
      }
      sub_39C2ED0((int *)a1, v4);
      v5 = **(__int64 ***)(a2 + 8);
      if ( v5 )
        v5 = (__int64 *)(*v5 & 0xFFFFFFFFFFFFFFF8LL);
      v10[0] = v5;
      v10[1] = 0;
      sub_39C5B00((int *)a1, v10);
      return;
    case 0xA:
      sub_39C2ED0((int *)a1, 0x41u);
      sub_39C2ED0((int *)a1, v3);
      sub_39C2ED0((int *)a1, 8u);
      v9 = *(_QWORD *)(a2 + 8);
      v7 = *(_QWORD *)(v9 + 8);
      v8 = *(int **)v9;
LABEL_12:
      sub_39C2EA0((int *)a1, v8, v7);
      return;
    default:
      return;
  }
}
