// Function: sub_33CC500
// Address: 0x33cc500
//
__int64 __fastcall sub_33CC500(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10)
{
  __int64 (*v12)(void); // rdx
  __int64 v13; // rax
  __int64 (*v14)(void); // rdx
  __int64 v15; // rax
  __int64 v16; // rax

  a1[5] = a2;
  a1[7] = a4;
  a1[12] = a3;
  v12 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 144LL);
  v13 = 0;
  if ( v12 != sub_2C8F680 )
  {
    v13 = v12();
    a2 = a1[5];
  }
  a1[2] = v13;
  v14 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 152LL);
  v15 = 0;
  if ( v14 != sub_30593D0 )
  {
    v15 = v14();
    a2 = a1[5];
  }
  a1[3] = a5;
  a1[1] = v15;
  v16 = sub_B2BE50(*(_QWORD *)a2);
  a1[10] = a6;
  a1[8] = v16;
  a1[13] = a7;
  a1[14] = a8;
  a1[15] = a9;
  a1[4] = a10;
  return a10;
}
