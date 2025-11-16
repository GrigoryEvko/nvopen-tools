// Function: sub_335E470
// Address: 0x335e470
//
void __fastcall sub_335E470(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9

  v3 = *a2;
  *(_QWORD *)a1 = a3;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = v3;
  *(_WORD *)(a1 + 24) = 0;
  sub_335E330(a1);
  sub_335E3B0(a1, (__int64)a2, v4, v5, v6, v7);
}
