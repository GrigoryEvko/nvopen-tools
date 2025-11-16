// Function: sub_2C2BB00
// Address: 0x2c2bb00
//
__int64 __fastcall sub_2C2BB00(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 result; // rax

  sub_C8CF70(a1, (void *)(a1 + 32), 8, a2 + 32, a2);
  v2 = *(_QWORD *)(a2 + 96);
  *(_QWORD *)(a2 + 96) = 0;
  *(_QWORD *)(a1 + 96) = v2;
  v3 = *(_QWORD *)(a2 + 104);
  *(_QWORD *)(a2 + 104) = 0;
  *(_QWORD *)(a1 + 104) = v3;
  v4 = *(_QWORD *)(a2 + 112);
  *(_QWORD *)(a2 + 112) = 0;
  *(_QWORD *)(a1 + 112) = v4;
  *(_WORD *)(a1 + 120) = *(_WORD *)(a2 + 120);
  sub_C8CF70(a1 + 128, (void *)(a1 + 160), 8, a2 + 160, a2 + 128);
  v5 = *(_QWORD *)(a2 + 224);
  *(_QWORD *)(a2 + 224) = 0;
  *(_QWORD *)(a1 + 224) = v5;
  v6 = *(_QWORD *)(a2 + 232);
  *(_QWORD *)(a2 + 232) = 0;
  *(_QWORD *)(a1 + 232) = v6;
  v7 = *(_QWORD *)(a2 + 240);
  *(_QWORD *)(a2 + 240) = 0;
  *(_QWORD *)(a1 + 240) = v7;
  result = *(unsigned __int16 *)(a2 + 248);
  *(_WORD *)(a1 + 248) = result;
  return result;
}
