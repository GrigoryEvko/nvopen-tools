// Function: sub_886F70
// Address: 0x886f70
//
void __fastcall sub_886F70(__int64 a1)
{
  _QWORD *v1; // r12
  int v2[3]; // [rsp+Ch] [rbp-14h] BYREF

  v2[0] = 0;
  v1 = qword_4D04998;
  qword_4D04998[6] = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 24) = v1;
  sub_885620((__int64)v1, dword_4F04C5C, v2);
  sub_881ED0((__int64)v1, dword_4F04C64, v2[0]);
}
