// Function: sub_289B350
// Address: 0x289b350
//
__int64 __fastcall sub_289B350(
        unsigned int ****a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  unsigned int v7; // r13d
  __int64 v8; // r9
  unsigned int ***v9; // rbx
  __int64 *v10; // rdi
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v18; // [rsp+8h] [rbp-D8h]
  __int64 v19; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v20; // [rsp+18h] [rbp-C8h]
  _QWORD v21[4]; // [rsp+30h] [rbp-B0h] BYREF
  const char *v22; // [rsp+50h] [rbp-90h] BYREF
  char v23; // [rsp+70h] [rbp-70h]
  char v24; // [rsp+71h] [rbp-6Fh]
  _QWORD v25[12]; // [rsp+80h] [rbp-60h] BYREF

  v7 = a3;
  v8 = *(_QWORD *)(a2 + 8);
  v9 = *a1;
  v24 = 1;
  v23 = 3;
  v10 = *(__int64 **)(v8 + 24);
  v20 = HIDWORD(a3);
  v11 = *(_QWORD *)(a5 + 8);
  v18 = v8;
  v22 = "mmul";
  v25[0] = a2;
  v19 = sub_BCDA70(v10, HIDWORD(a7) * (int)a3);
  v25[1] = a5;
  v12 = sub_BCB2D0((*v9)[9]);
  v25[2] = sub_ACD640(v12, v7, 0);
  v13 = sub_BCB2D0((*v9)[9]);
  v25[3] = sub_ACD640(v13, (unsigned int)v20, 0);
  v14 = sub_BCB2D0((*v9)[9]);
  v21[2] = v11;
  v25[4] = sub_ACD640(v14, HIDWORD(a7), 0);
  v21[0] = v19;
  v21[1] = v18;
  v15 = sub_B6E160(*(__int64 **)(*((_QWORD *)(*v9)[6] + 9) + 40LL), 0xE9u, (__int64)v21, 3);
  return sub_921880(*v9, *(_QWORD *)(v15 + 24), v15, (int)v25, 5, (__int64)&v22, 0);
}
