// Function: sub_23B40C0
// Address: 0x23b40c0
//
void __fastcall sub_23B40C0(__int64 a1, __int64 a2, char a3, char a4, unsigned int a5)
{
  __int64 v7; // rdx
  __int64 v9; // rdi
  bool v10; // zf
  _BOOL8 v11; // rsi
  bool v12; // r15
  __int64 v13; // rsi
  __int64 v14; // rax

  v7 = a1 + 24;
  v9 = a1 + 216;
  *(_QWORD *)(v9 - 208) = v7;
  *(_QWORD *)(v9 - 200) = 0x200000000LL;
  *(_BYTE *)(v9 - 7) = a5;
  *(_WORD *)(v9 - 6) = a5 >> 8;
  *(_DWORD *)(v9 - 16) = 0;
  *(_BYTE *)(v9 - 8) = a3;
  *(_DWORD *)(v9 - 4) = 0;
  sub_BC3B30(v9);
  nullsub_1492();
  *(_QWORD *)(a1 + 648) = a2;
  *(_QWORD *)(a1 + 664) = a1 + 680;
  *(_QWORD *)(a1 + 672) = 0x800000000LL;
  *(_BYTE *)(a1 + 641) = a3;
  v10 = unk_4F82DA8 == 1;
  *(_BYTE *)(a1 + 656) = 0;
  sub_23B3D30(a1 + 808, v10);
  *(_QWORD *)(a1 + 856) = 0;
  *(_QWORD *)(a1 + 864) = 0;
  *(_QWORD *)(a1 + 808) = &unk_4A15EF0;
  *(_QWORD *)(a1 + 872) = 0x4000000000LL;
  if ( unk_4F82DA8 == 5 )
  {
    v12 = 1;
    goto LABEL_5;
  }
  v11 = 1;
  v12 = unk_4F82DA8 == 6;
  if ( unk_4F82DA8 != 3 )
LABEL_5:
    v11 = unk_4F82DA8 == 5;
  sub_23B4070(a1 + 880, v11);
  *(_BYTE *)(a1 + 928) = v12;
  v10 = unk_4F82DA8 == 7;
  *(_QWORD *)(a1 + 880) = &unk_4A15FD0;
  sub_23B3670(a1 + 936, v10);
  *(_QWORD *)(a1 + 984) = a1 + 1000;
  sub_23AE760((__int64 *)(a1 + 984), "*** Dump of IR Before Last Pass Unknown ***", (__int64)"");
  sub_23B3D30(a1 + 1016, 1);
  *(_BYTE *)(a1 + 1064) = a3;
  v13 = (unsigned __int8)qword_4FDE5A8;
  *(_QWORD *)(a1 + 1016) = &unk_4A15F48;
  sub_3140B20(a1 + 1072, v13);
  *(_QWORD *)(a1 + 1072) = &unk_4A0A978;
  v14 = sub_C5F790(a1 + 1072, v13);
  sub_CE2060((_QWORD *)(a1 + 1232), v14);
  *(_BYTE *)(a1 + 1240) = a4;
}
