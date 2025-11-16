// Function: sub_3707C70
// Address: 0x3707c70
//
__int64 __fastcall sub_3707C70(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // r8
  __int64 v5; // r9

  v3 = a1 + 16;
  *(_QWORD *)(v3 - 8) = a2;
  *(_QWORD *)(v3 - 16) = &unk_4A3C938;
  sub_3708F20();
  *(_QWORD *)(a1 + 120) = a1 + 136;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_DWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = a1 + 88;
  *(_QWORD *)(a1 + 80) = 0x200000000LL;
  *(_QWORD *)(a1 + 128) = 0x200000000LL;
  return sub_C8D5F0(a1 + 72, (const void *)(a1 + 88), 0x1000u, 0x10u, v4, v5);
}
