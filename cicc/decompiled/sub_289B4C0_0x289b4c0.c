// Function: sub_289B4C0
// Address: 0x289b4c0
//
__int64 __fastcall sub_289B4C0(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        unsigned int a4,
        __int64 a5,
        unsigned int a6,
        __int64 a7,
        unsigned int ***a8,
        __int64 (__fastcall *a9)(__int64, __int64, unsigned __int64, _QWORD, __int64),
        __int64 a10)
{
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned int **v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // r14
  __int64 v29; // [rsp+18h] [rbp-C8h]
  __int64 v31; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v32; // [rsp+50h] [rbp-90h] BYREF
  int v33; // [rsp+58h] [rbp-88h]
  __int64 v34; // [rsp+60h] [rbp-80h] BYREF
  __int64 v35; // [rsp+68h] [rbp-78h]
  __int64 v36; // [rsp+70h] [rbp-70h]
  const char *v37; // [rsp+80h] [rbp-60h] BYREF
  __int64 v38; // [rsp+88h] [rbp-58h]
  char *v39; // [rsp+90h] [rbp-50h]
  __int16 v40; // [rsp+A0h] [rbp-40h]

  v37 = sub_BD5D20(a2);
  v13 = *(_QWORD *)(a2 + 8);
  v40 = 773;
  v39 = "_t";
  v38 = v14;
  v32 = sub_BCDA70(*(__int64 **)(v13 + 24), a4 * a3);
  v15 = *a8;
  v34 = a2;
  v16 = sub_BCB2D0(v15[9]);
  v35 = sub_ACD640(v16, a3, 0);
  v17 = sub_BCB2D0((*a8)[9]);
  v36 = sub_ACD640(v17, a4, 0);
  v18 = sub_B6E160(*(__int64 **)(*((_QWORD *)(*a8)[6] + 9) + 40LL), 0xEAu, (__int64)&v32, 1);
  v19 = sub_921880(*a8, *(_QWORD *)(v18 + 24), v18, (int)&v34, 3, (__int64)&v37, 0);
  v32 = __PAIR64__(a3, a4);
  v29 = v19;
  LOBYTE(v33) = dword_5003CC8 == 0;
  sub_2896BA0(a1, v19, __SPAIR64__(a3, a4), v33);
  v37 = sub_BD5D20(a5);
  v20 = *(_QWORD *)(a5 + 8);
  v40 = 773;
  v39 = "_t";
  v38 = v21;
  v22 = sub_BCDA70(*(__int64 **)(v20 + 24), (unsigned int)a7 * a6);
  v34 = a5;
  v31 = v22;
  v23 = sub_BCB2D0((*a8)[9]);
  v35 = sub_ACD640(v23, a6, 0);
  v24 = sub_BCB2D0((*a8)[9]);
  v36 = sub_ACD640(v24, (unsigned int)a7, 0);
  v25 = sub_B6E160(*(__int64 **)(*((_QWORD *)(*a8)[6] + 9) + 40LL), 0xEAu, (__int64)&v31, 1);
  v26 = sub_921880(*a8, *(_QWORD *)(v25 + 24), v25, (int)&v34, 3, (__int64)&v37, 0);
  v37 = (const char *)__PAIR64__(a6, a7);
  v27 = v26;
  LOBYTE(v38) = dword_5003CC8 == 0;
  sub_2896BA0(a1, v26, __SPAIR64__(a6, a7), v38);
  LOBYTE(v35) = dword_5003CC8 == 0;
  return a9(a10, v29, __PAIR64__(a3, a4), (unsigned int)v35, v27);
}
