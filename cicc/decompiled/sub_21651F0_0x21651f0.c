// Function: sub_21651F0
// Address: 0x21651f0
//
void *__fastcall sub_21651F0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, __int64 a5)
{
  __int64 v7; // rax

  sub_2164E40((_QWORD *)a1, a2, *(_QWORD *)a3, *(_QWORD *)(a3 + 8), *a4, a4[1]);
  *(_QWORD *)(a1 + 256) = a5;
  *(_QWORD *)(a1 + 224) = 0;
  *(_BYTE *)(a1 + 232) = 0;
  *(_QWORD *)a1 = &unk_4A02928;
  *(_QWORD *)(a1 + 216) = a1 + 232;
  *(_QWORD *)(a1 + 248) = 0x3400000000LL;
  sub_2162B60((_QWORD *)(a1 + 264));
  v7 = sub_2164EC0(a1, *(const char **)a3, *(_QWORD *)(a3 + 8));
  sub_21D9F20(a1 + 696, a5, v7);
  *(_QWORD *)(a1 + 82264) = &unk_4A3FED8;
  return sub_215CDD0(a1 + 82272, *(_DWORD *)(a2 + 44) == 23);
}
