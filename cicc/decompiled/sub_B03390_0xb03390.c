// Function: sub_B03390
// Address: 0xb03390
//
__int64 __fastcall sub_B03390(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4, char a5)
{
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rax

  v8 = sub_BCB2E0(a1);
  v9 = sub_ACD640(v8, a2, 1u);
  v12 = sub_B98A20(v9, a2, v10, v11);
  v13 = sub_BCB2E0(a1);
  v14 = sub_ACD640(v13, a3, 1u);
  v17 = sub_B98A20(v14, a3, v15, v16);
  return sub_B02F70(a1, v12, v17, 0, 0, a4, a5);
}
