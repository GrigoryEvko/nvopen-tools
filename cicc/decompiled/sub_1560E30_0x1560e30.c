// Function: sub_1560E30
// Address: 0x1560e30
//
__int64 __fastcall sub_1560E30(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // rax
  _QWORD *v4; // rax
  _QWORD *v5; // rax
  _QWORD *v6; // rax
  _QWORD *v7; // rax
  _QWORD *v8; // rax
  _QWORD *v9; // rax
  _QWORD *v10; // rax
  _QWORD *v11; // rax
  _QWORD *v13; // rax

  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = a1 + 16;
  *(_QWORD *)(a1 + 40) = a1 + 16;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  if ( *(_BYTE *)(a2 + 8) != 11 )
  {
    v13 = sub_15606E0((_QWORD *)a1, 40);
    sub_15606E0(v13, 58);
    if ( *(_BYTE *)(a2 + 8) == 15 )
      return a1;
  }
  v2 = sub_15606E0((_QWORD *)a1, 6);
  v3 = sub_15606E0(v2, 19);
  v4 = sub_15606E0(v3, 20);
  v5 = sub_15606E0(v4, 22);
  v6 = sub_15606E0(v5, 32);
  v7 = sub_1560C40(v6, 1);
  v8 = sub_1560C60(v7, 1);
  v9 = sub_15606E0(v8, 36);
  v10 = sub_15606E0(v9, 37);
  v11 = sub_15606E0(v10, 53);
  sub_15606E0(v11, 11);
  return a1;
}
