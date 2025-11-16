// Function: sub_128A3C0
// Address: 0x128a3c0
//
__int64 __fastcall sub_128A3C0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  _QWORD *v9; // [rsp+0h] [rbp-40h] BYREF
  __int64 v10; // [rsp+8h] [rbp-38h] BYREF
  char *v11; // [rsp+10h] [rbp-30h] BYREF
  char v12; // [rsp+20h] [rbp-20h]
  char v13; // [rsp+21h] [rbp-1Fh]

  v5 = *a2;
  v9 = a2;
  v10 = v5;
  v6 = sub_1644EA0(a3, &v10, 1, 0);
  v7 = sub_1632190(**(_QWORD **)(a1 + 32), *(_QWORD *)a4, *(unsigned int *)(a4 + 8), v6);
  v13 = 1;
  v12 = 3;
  v11 = "call";
  return sub_1285290((__int64 *)(a1 + 48), *(_QWORD *)(*(_QWORD *)v7 + 24LL), v7, (int)&v9, 1, (__int64)&v11, 0);
}
