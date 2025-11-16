// Function: sub_2FF19C0
// Address: 0x2ff19c0
//
__int64 __fastcall sub_2FF19C0(_BYTE *a1)
{
  void (*v1)(void); // rax
  _QWORD *v2; // rax
  __int64 result; // rax
  __int64 v4; // rax
  _QWORD *v5; // r12
  __int128 *v6; // rax
  _QWORD *v7; // rsi
  _QWORD *v8; // rax

  v1 = *(void (**)(void))(*(_QWORD *)a1 + 280LL);
  if ( (char *)v1 != (char *)sub_2FEDAB0 )
    v1();
  if ( !a1[276] )
  {
    if ( !(unsigned int)sub_2FF0570((__int64)a1) )
      goto LABEL_5;
    goto LABEL_10;
  }
  v4 = sub_22077B0(0xB0u);
  v5 = (_QWORD *)v4;
  if ( v4 )
  {
    *(_QWORD *)(v4 + 8) = 0;
    *(_DWORD *)(v4 + 24) = 3;
    *(_QWORD *)(v4 + 16) = &unk_502E0CC;
    *(_QWORD *)(v4 + 56) = v4 + 104;
    *(_QWORD *)(v4 + 112) = v4 + 160;
    *(_QWORD *)(v4 + 32) = 0;
    *(_DWORD *)(v4 + 88) = 1065353216;
    *(_QWORD *)(v4 + 40) = 0;
    *(_QWORD *)(v4 + 48) = 0;
    *(_QWORD *)(v4 + 64) = 1;
    *(_QWORD *)(v4 + 72) = 0;
    *(_QWORD *)(v4 + 80) = 0;
    *(_QWORD *)(v4 + 96) = 0;
    *(_QWORD *)(v4 + 104) = 0;
    *(_QWORD *)(v4 + 120) = 1;
    *(_QWORD *)(v4 + 128) = 0;
    *(_QWORD *)(v4 + 136) = 0;
    *(_QWORD *)(v4 + 152) = 0;
    *(_QWORD *)(v4 + 160) = 0;
    *(_BYTE *)(v4 + 168) = 0;
    *(_QWORD *)v4 = &unk_4A2D3B0;
    *(_DWORD *)(v4 + 144) = 1065353216;
    v6 = sub_BC2B00();
    sub_30A47F0(v6);
  }
  sub_2FF0E80((__int64)a1, v5, 0);
  if ( (unsigned int)sub_2FF0570((__int64)a1) )
  {
LABEL_10:
    v7 = (_QWORD *)sub_270B4D0();
    sub_2FF0E80((__int64)a1, v7, 0);
  }
LABEL_5:
  v2 = (_QWORD *)sub_2D556D0();
  result = sub_2FF0E80((__int64)a1, v2, 0);
  if ( !a1[273] )
  {
    v8 = (_QWORD *)sub_BE0980(1);
    return sub_2FF0E80((__int64)a1, v8, 0);
  }
  return result;
}
