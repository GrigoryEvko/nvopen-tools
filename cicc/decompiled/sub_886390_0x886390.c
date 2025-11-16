// Function: sub_886390
// Address: 0x886390
//
void __fastcall sub_886390(__int64 a1)
{
  __int64 v1; // r13
  _QWORD *v2; // rax
  __int64 v3; // r12
  int v4; // esi
  int v5[9]; // [rsp+Ch] [rbp-24h] BYREF

  v5[0] = 0;
  v1 = *(_QWORD *)(a1 + 88);
  if ( !sub_879510((_QWORD *)a1) && (*(_BYTE *)(a1 + 81) & 0x20) == 0 )
  {
    v2 = sub_87EBB0(3u, *(_QWORD *)a1, (_QWORD *)(a1 + 48));
    *((_BYTE *)v2 + 81) |= 0x10u;
    v3 = (__int64)v2;
    v2[11] = v1;
    v4 = dword_4F04C64;
    v2[8] = v1;
    *((_BYTE *)v2 + 104) = 1;
    sub_885620((__int64)v2, v4, v5);
    sub_881ED0(v3, dword_4F04C64, v5[0]);
  }
}
