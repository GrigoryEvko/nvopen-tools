// Function: sub_74C5E0
// Address: 0x74c5e0
//
void __fastcall sub_74C5E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, char a7, __int64 a8)
{
  __int64 v9; // r13
  char v10; // al
  void (__fastcall *v11)(char *, __int64); // r13
  char *v12; // rdi
  void (__fastcall *v13)(char *, __int64); // rbx
  char *v14; // rax
  void (__fastcall *v15)(char *, __int64); // rbx
  char *v16; // rax
  void (__fastcall *v17)(char *, __int64); // rbx
  char *v18; // rax
  void (__fastcall *v19)(char *, __int64); // rbx
  char *v20; // rax
  void (__fastcall *v21)(char *, __int64); // rbx
  int v22; // edi
  char *v23; // rax
  void (__fastcall *v24)(char *, __int64); // rbx
  char *v25; // rax
  void (__fastcall *v26)(char *, __int64); // rbx
  char *v27; // rax
  void (__fastcall *v28)(char *, __int64); // rbx
  char *v29; // rdi

  if ( *(_BYTE *)(a1 + 136) )
  {
    (*(void (__fastcall **)(const char *, __int64))a1)("(decltype(^0){})", a1);
  }
  else
  {
    v9 = a8;
    if ( a7 == 48 )
    {
      v10 = *(_BYTE *)(a8 + 8);
      if ( v10 == 1 )
      {
        v9 = *(_QWORD *)(a8 + 32);
LABEL_19:
        sub_748000(v9, 0, a1, a4, a5);
      }
      else if ( v10 == 2 )
      {
        v9 = *(_QWORD *)(a8 + 32);
LABEL_16:
        v15 = *(void (__fastcall **)(char *, __int64))a1;
        v16 = sub_67C860(1470);
        v15(v16, a1);
        (*(void (__fastcall **)(char *, __int64))a1)(" ", a1);
        sub_74C550(v9, 0x3Bu, a1);
      }
      else
      {
        if ( v10 )
          sub_721090();
        v9 = *(_QWORD *)(a8 + 32);
LABEL_13:
        v13 = *(void (__fastcall **)(char *, __int64))a1;
        v14 = sub_67C860(1465);
        v13(v14, a1);
        (*(void (__fastcall **)(char *, __int64))a1)(" ", a1);
        sub_74B930(v9, a1);
      }
    }
    else
    {
      switch ( a7 )
      {
        case 0:
          v28 = *(void (__fastcall **)(char *, __int64))a1;
          v29 = sub_67C860(3379);
          v28(v29, a1);
          return;
        case 2:
          goto LABEL_19;
        case 6:
          goto LABEL_13;
        case 7:
          v21 = *(void (__fastcall **)(char *, __int64))a1;
          v22 = 1475;
          goto LABEL_23;
        case 8:
          v19 = *(void (__fastcall **)(char *, __int64))a1;
          v20 = sub_67C860(1481);
          v19(v20, a1);
          (*(void (__fastcall **)(char *, __int64))a1)(" ", a1);
          sub_74C550(a8, 8u, a1);
          return;
        case 11:
          v21 = *(void (__fastcall **)(char *, __int64))a1;
          v22 = 1478;
LABEL_23:
          v23 = sub_67C860(v22);
          v21(v23, a1);
          (*(void (__fastcall **)(char *, __int64))a1)(" ", a1);
          sub_74C550(a8, 0xBu, a1);
          return;
        case 13:
          sub_747C50(a8, a1);
          return;
        case 23:
          if ( (unsigned __int8)(*(_BYTE *)(a8 + 28) - 3) > 1u )
            goto LABEL_11;
          v26 = *(void (__fastcall **)(char *, __int64))a1;
          v27 = sub_67C860(1482);
          v26(v27, a1);
          (*(void (__fastcall **)(char *, __int64))a1)(" ", a1);
          sub_74C550(*(_QWORD *)(a8 + 32), 0x1Cu, a1);
          break;
        case 28:
          v24 = *(void (__fastcall **)(char *, __int64))a1;
          v25 = sub_67C860(3380);
          v24(v25, a1);
          (*(void (__fastcall **)(char *, __int64))a1)(" ", a1);
          sub_74C550(a8, 0x1Cu, a1);
          return;
        case 37:
          sub_74B930(*(_QWORD *)(a8 + 40), a1);
          (*(void (__fastcall **)(char *, __int64))a1)(" ", a1);
          v17 = *(void (__fastcall **)(char *, __int64))a1;
          v18 = sub_67C860(2246);
          v17(v18, a1);
          (*(void (__fastcall **)(char *, __int64))a1)(" ", a1);
          sub_74B930(*(_QWORD *)(a8 + 56), a1);
          return;
        case 59:
          goto LABEL_16;
        default:
LABEL_11:
          v11 = *(void (__fastcall **)(char *, __int64))a1;
          v12 = sub_67C860(3381);
          v11(v12, a1);
          break;
      }
    }
  }
}
