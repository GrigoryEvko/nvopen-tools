// Function: sub_1E2CD10
// Address: 0x1e2cd10
//
void __fastcall sub_1E2CD10(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 (*v3)(); // rax
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rax

  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 5;
  *(_QWORD *)(a1 + 16) = &unk_4FC6A0E;
  *(_QWORD *)(a1 + 80) = a1 + 64;
  *(_QWORD *)(a1 + 88) = a1 + 64;
  *(_QWORD *)(a1 + 128) = a1 + 112;
  *(_QWORD *)(a1 + 136) = a1 + 112;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_BYTE *)(a1 + 152) = 0;
  *(_QWORD *)a1 = &unk_49FBE00;
  v2 = *a2;
  *(_QWORD *)(a1 + 160) = a2;
  v3 = *(__int64 (**)())(v2 + 24);
  if ( v3 == sub_16FF760 || (v6 = ((__int64 (__fastcall *)(__int64 *))v3)(a2)) == 0 )
    v4 = 0;
  else
    v4 = v6 + 8;
  sub_38BCD70(a1 + 168, a2[76], a2[77], v4, 0, 0);
  *(_QWORD *)(a1 + 1704) = 0;
  *(_QWORD *)(a1 + 1712) = 0;
  *(_QWORD *)(a1 + 1720) = 0;
  *(_QWORD *)(a1 + 1752) = 0;
  *(_QWORD *)(a1 + 1760) = 0;
  *(_QWORD *)(a1 + 1768) = 0;
  *(_DWORD *)(a1 + 1776) = 0;
  *(_DWORD *)(a1 + 1784) = 0;
  *(_QWORD *)(a1 + 1792) = 0;
  *(_QWORD *)(a1 + 1800) = 0;
  v5 = sub_163A1D0();
  sub_1E2CB50(v5);
}
