// Function: sub_3717640
// Address: 0x3717640
//
unsigned __int64 __fastcall sub_3717640(__int64 **a1)
{
  __int64 *v1; // r12
  unsigned __int64 result; // rax
  __int64 v3; // rbx
  __int64 v4; // r13
  __int64 v5; // r14
  __int64 v6; // r15
  __int64 v7; // [rsp+0h] [rbp-A0h]
  __int64 v8; // [rsp+8h] [rbp-98h]
  __int64 v9; // [rsp+10h] [rbp-90h]
  __int64 v10[16]; // [rsp+20h] [rbp-80h] BYREF

  v1 = *a1;
  result = sub_BCBBB0(*a1, (__int64)"struct.__tgt_offload_entry", 26);
  if ( !result )
  {
    v3 = sub_BCE3C0(v1, 0);
    v4 = sub_BCB2E0(v1);
    v5 = sub_BCB2E0(v1);
    v6 = sub_BCE3C0(v1, 0);
    v7 = sub_BCE3C0(v1, 0);
    v8 = sub_BCB2D0(v1);
    v9 = sub_BCB2C0(v1);
    v10[5] = v6;
    v10[1] = sub_BCB2C0(v1);
    v10[2] = v9;
    v10[3] = v8;
    v10[4] = v7;
    v10[6] = v5;
    v10[7] = v4;
    v10[8] = v3;
    v10[0] = sub_BCB2E0(v1);
    return sub_BD0EC0(v10, 9, "struct.__tgt_offload_entry", 0x1Au, 0);
  }
  return result;
}
