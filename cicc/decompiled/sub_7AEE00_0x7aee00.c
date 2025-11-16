// Function: sub_7AEE00
// Address: 0x7aee00
//
__int64 __fastcall sub_7AEE00(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // rax

  v6 = (__int64)qword_4F06430;
  if ( qword_4F06430 )
    qword_4F06430 = (_QWORD *)*qword_4F06430;
  else
    v6 = sub_823970(112);
  *(_QWORD *)(v6 + 8) = 0;
  *(_WORD *)(v6 + 48) &= 0xF800u;
  v7 = unk_4F06440;
  *(_QWORD *)(v6 + 16) = a1;
  *(_QWORD *)(v6 + 24) = 0;
  *(_QWORD *)v6 = v7;
  v8 = unk_4F06428;
  *(_QWORD *)(v6 + 32) = a2;
  *(_QWORD *)(v6 + 56) = a3;
  unk_4F06428 = v8 + 1;
  *(_QWORD *)(v6 + 80) = v8 + 1;
  *(_QWORD *)(v6 + 40) = 0;
  *(_QWORD *)(v6 + 64) = a4;
  *(_QWORD *)(v6 + 72) = 0;
  *(_DWORD *)(v6 + 88) = 0;
  *(_WORD *)(v6 + 92) = 0;
  *(_QWORD *)(v6 + 96) = 0;
  *(_QWORD *)(v6 + 104) = 0;
  if ( a1 )
  {
    *(_BYTE *)(v6 + 50) = *a1;
    *a1 = 10;
  }
  else
  {
    *(_BYTE *)(v6 + 50) = 32;
    unk_4F06438 = v6;
  }
  unk_4F06440 = v6;
  dword_4F17FA0 = 0;
  sub_7AED40(v6);
  return v6;
}
