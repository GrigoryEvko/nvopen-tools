// Function: sub_3733B80
// Address: 0x3733b80
//
void __fastcall sub_3733B80(__int64 a1, __int64 a2, unsigned __int16 a3)
{
  unsigned __int16 v3; // r13
  __int64 **v4; // r14
  unsigned __int64 v5; // rax
  unsigned int v6; // eax
  _QWORD *v7; // rax
  __int64 v8; // rax
  size_t v9; // rdx
  int *v10; // rsi
  __int64 v11; // rsi
  size_t *v12; // rsi
  int *v13; // r8
  unsigned __int64 v14; // rsi
  __int64 **v15; // r14
  unsigned __int64 v16; // rax
  unsigned __int64 v17[6]; // [rsp+0h] [rbp-30h] BYREF

  v3 = *(_WORD *)(a2 + 4);
  switch ( *(_DWORD *)a2 )
  {
    case 0:
    case 3:
    case 4:
    case 5:
    case 6:
    case 0xC:
LABEL_20:
      BUG();
    case 1:
      sub_372FCB0((int *)a1, 0x41u);
      sub_372FCB0((int *)a1, v3);
      switch ( *(_WORD *)(a2 + 6) )
      {
        case 5:
        case 6:
        case 7:
        case 0xB:
        case 0xD:
        case 0xF:
          sub_372FCB0((int *)a1, 0xDu);
          sub_372FD30((int *)a1, *(_QWORD *)(a2 + 8));
          return;
        case 0xC:
        case 0x19:
          sub_372FCB0((int *)a1, 0xCu);
          sub_372FCB0((int *)a1, *(_QWORD *)(a2 + 8));
          return;
        default:
          goto LABEL_20;
      }
    case 2:
      sub_372FCB0((int *)a1, 0x41u);
      sub_372FCB0((int *)a1, v3);
      sub_372FCB0((int *)a1, 8u);
      v11 = *(_QWORD *)(a2 + 8);
      if ( (v11 & 4) != 0 )
      {
        v14 = v11 & 0xFFFFFFFFFFFFFFF8LL;
        v13 = *(int **)(v14 + 24);
        v9 = *(_QWORD *)(v14 + 32);
      }
      else
      {
        v12 = (size_t *)(v11 & 0xFFFFFFFFFFFFFFF8LL);
        v9 = *v12;
        v13 = (int *)(v12 + 4);
      }
      v10 = v13;
      goto LABEL_10;
    case 7:
      sub_3733850(a1, v3, a3, *(_QWORD *)(a2 + 8));
      return;
    case 8:
    case 9:
    case 0xA:
      sub_372FCB0((int *)a1, 0x41u);
      sub_372FCB0((int *)a1, v3);
      sub_372FCB0((int *)a1, 9u);
      if ( *(_DWORD *)a2 == 8 )
      {
        v15 = *(__int64 ***)(a2 + 8);
        v16 = sub_31DF6E0(*(_QWORD *)(a1 + 152));
        LODWORD(v17[0]) = v16;
        WORD2(v17[0]) = WORD2(v16);
        v6 = sub_3215F30(v15, (__int64)v17);
      }
      else
      {
        if ( *(_DWORD *)a2 != 9 )
        {
          sub_37332D0(a1, (__int64 *)(a2 + 8));
          return;
        }
        v4 = *(__int64 ***)(a2 + 8);
        v5 = sub_31DF6E0(*(_QWORD *)(a1 + 152));
        LODWORD(v17[0]) = v5;
        WORD2(v17[0]) = WORD2(v5);
        v6 = sub_3215EC0(v4, (__int64)v17);
      }
      sub_372FCB0((int *)a1, v6);
      v7 = **(_QWORD ***)(a2 + 8);
      if ( v7 )
        v7 = (_QWORD *)(*v7 & 0xFFFFFFFFFFFFFFF8LL);
      v17[0] = (unsigned __int64)v7;
      v17[1] = 0;
      sub_37333F0(a1, v17);
      return;
    case 0xB:
      sub_372FCB0((int *)a1, 0x41u);
      sub_372FCB0((int *)a1, v3);
      sub_372FCB0((int *)a1, 8u);
      v8 = *(_QWORD *)(a2 + 8);
      v9 = *(_QWORD *)(v8 + 8);
      v10 = *(int **)v8;
LABEL_10:
      sub_372FC80((int *)a1, v10, v9);
      return;
    default:
      return;
  }
}
