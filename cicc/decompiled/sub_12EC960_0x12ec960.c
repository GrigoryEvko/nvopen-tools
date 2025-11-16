// Function: sub_12EC960
// Address: 0x12ec960
//
__int64 __fastcall sub_12EC960(__int64 a1, const void *a2, size_t a3)
{
  int v4; // eax

  sub_16913D0(a1 + 8, &unk_497C3C0, 221, 0);
  *(_QWORD *)a1 = &unk_49E7460;
  v4 = sub_12EC8D0(a1, a2, a3);
  *(_QWORD *)(a1 + 272) = 0;
  *(_DWORD *)(a1 + 112) = v4;
  *(_QWORD *)(a1 + 128) = a1 + 144;
  *(_QWORD *)(a1 + 136) = 0x1000000000LL;
  *(_QWORD *)(a1 + 120) = &unk_49EE9E8;
  *(_QWORD *)(a1 + 312) = 0x1000000000LL;
  *(_QWORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 288) = 0;
  *(_DWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 304) = a1 + 320;
  *(_QWORD *)(a1 + 456) = a1 + 448;
  *(_QWORD *)(a1 + 448) = a1 + 448;
  *(_QWORD *)(a1 + 464) = 0;
  *(_DWORD *)(a1 + 472) = 0;
  return a1 + 448;
}
