// Function: sub_1407870
// Address: 0x1407870
//
__int64 *__fastcall sub_1407870(__int64 a1, __int64 a2)
{
  __int64 *v2; // r13
  __int64 v5; // r8
  __int64 v6; // rax
  _QWORD *v7; // r14
  __int64 v8; // r10
  __int64 v9; // r9
  __int64 v10; // rsi
  __int64 v11; // rdi
  _QWORD *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 *result; // rax
  __int64 v16; // rcx
  __int64 *v17; // rax
  __int64 v18; // r14
  __int64 *v19; // rax
  __int64 **v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // [rsp+0h] [rbp-D0h]
  __int64 v23; // [rsp+18h] [rbp-B8h] BYREF
  __int64 v24[3]; // [rsp+20h] [rbp-B0h] BYREF
  _QWORD *v25; // [rsp+38h] [rbp-98h]
  __int64 v26[4]; // [rsp+40h] [rbp-90h] BYREF
  __int64 v27; // [rsp+60h] [rbp-70h] BYREF
  __int64 v28; // [rsp+68h] [rbp-68h]
  __int64 v29; // [rsp+70h] [rbp-60h]
  _QWORD *v30; // [rsp+78h] [rbp-58h]
  __int64 v31; // [rsp+80h] [rbp-50h] BYREF
  __int64 v32; // [rsp+88h] [rbp-48h]
  __int64 v33; // [rsp+90h] [rbp-40h]
  _QWORD *v34; // [rsp+98h] [rbp-38h]

  v2 = (__int64 *)(a1 + 568);
  v5 = *(_QWORD *)(a1 + 584);
  v6 = *(_QWORD *)(a1 + 616);
  v23 = a2;
  v7 = *(_QWORD **)(a1 + 640);
  v8 = *(_QWORD *)(a1 + 624);
  v9 = *(_QWORD *)(a1 + 632);
  v10 = *(_QWORD *)(a1 + 600);
  v27 = v5;
  v11 = *(_QWORD *)(a1 + 592);
  v12 = *(_QWORD **)(a1 + 608);
  v31 = v6;
  v29 = v10;
  v28 = v11;
  v30 = v12;
  v22 = v6;
  v32 = v8;
  v33 = v9;
  v34 = v7;
  sub_1404960(v24, (__int64)&v27, &v31, &v23);
  v27 = v22;
  v13 = *v7;
  v30 = v7;
  v28 = v13;
  v29 = v13 + 512;
  v31 = v24[0];
  v14 = *v25;
  v34 = v25;
  v32 = v14;
  v33 = v14 + 512;
  result = sub_14072B0(v26, v2, &v31, &v27);
  if ( *(_QWORD *)(a1 + 656) == a2 )
  {
    v16 = *(_QWORD *)(a1 + 632);
    v17 = *(__int64 **)(a1 + 616);
    *(_BYTE *)(a1 + 664) = 1;
    if ( v17 == (__int64 *)(v16 - 8) )
    {
      v18 = *(_QWORD *)(a1 + 640);
      if ( (((__int64)v17 - *(_QWORD *)(a1 + 624)) >> 3)
         + ((((v18 - *(_QWORD *)(a1 + 608)) >> 3) - 1) << 6)
         + ((__int64)(*(_QWORD *)(a1 + 600) - *(_QWORD *)(a1 + 584)) >> 3) == 0xFFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
      if ( (unsigned __int64)(*(_QWORD *)(a1 + 576) - ((v18 - *(_QWORD *)(a1 + 568)) >> 3)) <= 1 )
      {
        sub_1404C40(v2, 1u, 0);
        v18 = *(_QWORD *)(a1 + 640);
      }
      *(_QWORD *)(v18 + 8) = sub_22077B0(512);
      v19 = *(__int64 **)(a1 + 616);
      if ( v19 )
        *v19 = a2;
      v20 = (__int64 **)(*(_QWORD *)(a1 + 640) + 8LL);
      *(_QWORD *)(a1 + 640) = v20;
      result = *v20;
      v21 = (__int64)(*v20 + 64);
      *(_QWORD *)(a1 + 624) = result;
      *(_QWORD *)(a1 + 632) = v21;
      *(_QWORD *)(a1 + 616) = result;
    }
    else
    {
      if ( v17 )
      {
        *v17 = a2;
        v17 = *(__int64 **)(a1 + 616);
      }
      result = v17 + 1;
      *(_QWORD *)(a1 + 616) = result;
    }
  }
  return result;
}
