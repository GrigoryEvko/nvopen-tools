// Function: sub_92BCC0
// Address: 0x92bcc0
//
__int64 __fastcall sub_92BCC0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int64 v5; // rax
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  int v8; // edx
  __int64 v10; // [rsp+0h] [rbp-50h] BYREF
  __int64 v11; // [rsp+8h] [rbp-48h] BYREF
  char *v12; // [rsp+10h] [rbp-40h] BYREF
  char v13; // [rsp+30h] [rbp-20h]
  char v14; // [rsp+31h] [rbp-1Fh]

  v5 = *(_QWORD *)(a2 + 8);
  v10 = a2;
  v11 = v5;
  v6 = sub_BCF480(a3, &v11, 1, 0);
  v7 = sub_BA8CA0(**(_QWORD **)(a1 + 32), *a4, a4[1], v6);
  v14 = 1;
  v12 = "call";
  v13 = 3;
  return sub_921880((unsigned int **)(a1 + 48), v7, v8, (int)&v10, 1, (__int64)&v12, 0);
}
