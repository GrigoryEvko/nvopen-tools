// Function: sub_1FEB540
// Address: 0x1feb540
//
__int64 __fastcall sub_1FEB540(
        __int64 a1,
        unsigned int *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        double a6,
        double a7,
        double a8)
{
  __int64 *v11; // rdi
  __int64 v12; // rsi
  __int64 v13; // r10
  __int64 v14; // rax
  _QWORD *v15; // rdi
  __int64 v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // r9
  __int64 v19; // rdx
  __int64 v21; // [rsp-50h] [rbp-90h]
  __int64 v22; // [rsp-50h] [rbp-90h]
  __int64 v23; // [rsp-48h] [rbp-88h]
  __int64 v24; // [rsp-48h] [rbp-88h]
  __int128 v25; // [rsp-40h] [rbp-80h]
  __int128 v26; // [rsp-40h] [rbp-80h]
  __int64 v27; // [rsp-30h] [rbp-70h]
  __int64 v28; // [rsp-30h] [rbp-70h]
  __int128 v29; // [rsp-10h] [rbp-50h]
  __int64 v30; // [rsp+0h] [rbp-40h] BYREF
  __int64 v31; // [rsp+8h] [rbp-38h]
  __int64 v32; // [rsp+10h] [rbp-30h]

  v11 = *(__int64 **)(a1 + 16);
  if ( *((_QWORD *)a2 + 2) )
  {
    v12 = *((_QWORD *)a2 + 2);
    v13 = *((_QWORD *)a2 + 3);
    v27 = *((_QWORD *)a2 + 10);
    v25 = *((_OWORD *)a2 + 4);
    v23 = *((_QWORD *)a2 + 7);
    v21 = *((_QWORD *)a2 + 6);
    v30 = 0;
    v31 = 0;
    v32 = 0;
    v14 = sub_1D2C750(v11, v12, v13, a3, a4, a5, v21, v23, v25, v27, 3, 0, 0, 0, (__int64)&v30);
    v15 = *(_QWORD **)(a1 + 16);
    v16 = *a2;
    v18 = v17;
    v19 = *((_QWORD *)a2 + 1);
    v28 = *((_QWORD *)a2 + 13);
    v26 = *(_OWORD *)(a2 + 22);
    v24 = *((_QWORD *)a2 + 5);
    v22 = *((_QWORD *)a2 + 4);
    v30 = 0;
    v31 = 0;
    v32 = 0;
    return sub_1D2B730(v15, v16, v19, a3, v14, v18, v22, v24, v26, v28, 0, 0, (__int64)&v30, 0);
  }
  else
  {
    *((_QWORD *)&v29 + 1) = a5;
    *(_QWORD *)&v29 = a4;
    return sub_1D309E0(v11, 158, a3, *a2, *((const void ***)a2 + 1), 0, a6, a7, a8, v29);
  }
}
