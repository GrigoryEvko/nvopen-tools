// Function: sub_2EAA0B0
// Address: 0x2eaa0b0
//
void __fastcall sub_2EAA0B0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // rax
  __int64 (*v6)(); // rdx
  __int64 v7; // rax

  v2 = a2[82];
  v3 = a2[83];
  v4 = a2[85];
  *(_QWORD *)a1 = a2;
  sub_E64450(a1 + 8, (__int64)(a2 + 64), v2, v3, v4, 0, (__int64)(a2 + 122), 0, 0);
  *(_QWORD *)(a1 + 2504) = 0;
  v5 = *a2;
  *(_QWORD *)(a1 + 2480) = 0;
  *(_QWORD *)(a1 + 2488) = 0;
  *(_QWORD *)(a1 + 2512) = 0;
  *(_QWORD *)(a1 + 2520) = 0;
  *(_DWORD *)(a1 + 2528) = 0;
  *(_DWORD *)(a1 + 2536) = 0;
  *(_QWORD *)(a1 + 2544) = 0;
  *(_QWORD *)(a1 + 2552) = 0;
  v6 = *(__int64 (**)())(v5 + 24);
  v7 = 0;
  if ( v6 != sub_23CE280 )
    v7 = ((__int64 (__fastcall *)(__int64 *))v6)(a2);
  *(_QWORD *)(a1 + 176) = v7;
  sub_2EA9F00(a1);
}
