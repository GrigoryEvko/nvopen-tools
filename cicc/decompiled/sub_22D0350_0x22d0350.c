// Function: sub_22D0350
// Address: 0x22d0350
//
__int64 __fastcall sub_22D0350(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v5; // rdx

  v4 = sub_BC1CD0(a4, &unk_4F875F0, a3);
  v5 = *a2;
  *(_BYTE *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = v4 + 8;
  *(_QWORD *)a1 = v5;
  return a1;
}
