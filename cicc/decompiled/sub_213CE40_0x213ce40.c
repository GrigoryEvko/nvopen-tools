// Function: sub_213CE40
// Address: 0x213ce40
//
__int64 *__fastcall sub_213CE40(__int64 *a1, __int64 a2, int a3, double a4, double a5, double a6)
{
  __int64 *v7; // r14
  __int64 v8; // r15
  __int64 (__fastcall *v9)(__int64, __int64); // rbx
  __int64 v10; // rax
  unsigned int v11; // edx
  unsigned __int8 v12; // al
  __int64 v13; // rsi
  unsigned int v14; // r15d
  __int64 v15; // r14
  __int64 v16; // rdx
  __int64 v17; // r15
  __int64 v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int128 v22; // [rsp-10h] [rbp-50h]
  __int64 v23; // [rsp+0h] [rbp-40h] BYREF
  int v24; // [rsp+8h] [rbp-38h]

  v7 = (__int64 *)a1[1];
  if ( a3 == 1 )
  {
    v19 = *(_QWORD *)(a2 + 32);
    v20 = sub_2138AD0((__int64)a1, *(_QWORD *)(v19 + 40), *(_QWORD *)(v19 + 48));
    return sub_1D2E2F0(
             v7,
             (__int64 *)a2,
             **(_QWORD **)(a2 + 32),
             *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
             v20,
             v21,
             *(_OWORD *)(v19 + 80));
  }
  else
  {
    v8 = *a1;
    v9 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)*a1 + 48LL);
    v10 = sub_1E0A0C0(v7[4]);
    if ( v9 == sub_1D13A20 )
    {
      v11 = 8 * sub_15A9520(v10, 0);
      if ( v11 == 32 )
      {
        v12 = 5;
      }
      else if ( v11 > 0x20 )
      {
        v12 = 6;
        if ( v11 != 64 )
        {
          v12 = 0;
          if ( v11 == 128 )
            v12 = 7;
        }
      }
      else
      {
        v12 = 3;
        if ( v11 != 8 )
          v12 = 4 * (v11 == 16);
      }
    }
    else
    {
      v12 = v9(v8, v10);
    }
    v13 = *(_QWORD *)(a2 + 72);
    v14 = v12;
    v23 = v13;
    if ( v13 )
      sub_1623A60((__int64)&v23, v13, 2);
    v24 = *(_DWORD *)(a2 + 64);
    v15 = sub_1D323C0(
            v7,
            *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL),
            *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL),
            (__int64)&v23,
            v14,
            0,
            a4,
            a5,
            a6);
    v17 = v16;
    if ( v23 )
      sub_161E7C0((__int64)&v23, v23);
    *((_QWORD *)&v22 + 1) = v17;
    *(_QWORD *)&v22 = v15;
    return sub_1D2E2F0(
             (_QWORD *)a1[1],
             (__int64 *)a2,
             **(_QWORD **)(a2 + 32),
             *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
             *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
             *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL),
             v22);
  }
}
