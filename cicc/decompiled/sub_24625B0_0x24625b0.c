// Function: sub_24625B0
// Address: 0x24625b0
//
_QWORD *__fastcall sub_24625B0(__int64 *a1)
{
  _QWORD *v1; // r13
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // r14
  _QWORD *v5; // r12
  __int64 v7; // [rsp+8h] [rbp-58h]
  const char *v8; // [rsp+10h] [rbp-50h] BYREF
  char v9; // [rsp+30h] [rbp-30h]
  char v10; // [rsp+31h] [rbp-2Fh]

  v1 = (_QWORD *)sub_BCB2D0(*(_QWORD **)(a1[1] + 72));
  v2 = *(unsigned __int8 *)(a1[2] + 8);
  v3 = sub_BCB2D0(*(_QWORD **)(a1[1] + 72));
  v10 = 1;
  v4 = sub_ACD640(v3, v2, 0);
  v9 = 3;
  v8 = "__msan_keep_going";
  BYTE4(v7) = 0;
  v5 = sub_BD2C40(88, unk_3F0FAE8);
  if ( v5 )
    sub_B30000((__int64)v5, *a1, v1, 1, 5, v4, (__int64)&v8, 0, 0, v7, 0);
  return v5;
}
