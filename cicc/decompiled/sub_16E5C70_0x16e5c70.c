// Function: sub_16E5C70
// Address: 0x16e5c70
//
__int64 __fastcall sub_16E5C70(int *a1, __int64 a2, __int64 a3)
{
  int v3; // eax
  _QWORD v5[2]; // [rsp+0h] [rbp-20h] BYREF
  int v6; // [rsp+10h] [rbp-10h]

  v3 = *a1;
  v5[1] = "0x%08X";
  v6 = v3;
  v5[0] = &unk_49EFAC8;
  return sub_16E8450(a3, v5);
}
