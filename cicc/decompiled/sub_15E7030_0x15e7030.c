// Function: sub_15E7030
// Address: 0x15e7030
//
_QWORD *__fastcall sub_15E7030(__int64 *a1, int a2, __int64 *a3)
{
  __int64 v4; // rax
  __int64 *v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 *v10; // [rsp+8h] [rbp-48h] BYREF
  __int64 v11[2]; // [rsp+10h] [rbp-40h] BYREF
  char v12; // [rsp+20h] [rbp-30h] BYREF
  __int16 v13; // [rsp+30h] [rbp-20h]

  v4 = a1[1];
  v10 = a3;
  v5 = *(__int64 **)(*(_QWORD *)(v4 + 56) + 40LL);
  v6 = *a3;
  v7 = **(_QWORD **)(*a3 + 16);
  v11[1] = v6;
  v11[0] = v7;
  v8 = sub_15E26F0(v5, a2, v11, 2);
  v13 = 257;
  return sub_15E6DE0(v8, (int)&v10, 1, a1, (int)&v12, 0, 0, 0);
}
