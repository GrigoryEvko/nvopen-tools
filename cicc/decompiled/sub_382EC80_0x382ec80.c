// Function: sub_382EC80
// Address: 0x382ec80
//
__int64 *__fastcall sub_382EC80(__int64 *a1, __int64 a2, int a3, __m128i a4)
{
  __int64 v6; // r14
  __int64 v7; // r15
  __int64 (__fastcall *v8)(__int64, __int64, __int64, __int64); // rbx
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 (__fastcall *v11)(__int64, __int64, unsigned int); // rax
  int v12; // edx
  unsigned __int16 v13; // ax
  __int64 v14; // rsi
  unsigned int v15; // r15d
  unsigned __int8 *v16; // r14
  __int64 v17; // rdx
  __int64 v18; // r15
  __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int128 v23; // [rsp-10h] [rbp-50h]
  __int64 v24; // [rsp+0h] [rbp-40h] BYREF
  int v25; // [rsp+8h] [rbp-38h]

  v6 = a1[1];
  if ( a3 == 1 )
  {
    v20 = *(_QWORD *)(a2 + 40);
    v21 = sub_37AE0F0((__int64)a1, *(_QWORD *)(v20 + 40), *(_QWORD *)(v20 + 48));
    return sub_33EC3B0(
             (_QWORD *)v6,
             (__int64 *)a2,
             **(_QWORD **)(a2 + 40),
             *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
             v21,
             v22,
             *(_OWORD *)(v20 + 80));
  }
  else
  {
    v7 = *a1;
    v8 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)*a1 + 72LL);
    v9 = sub_2E79000(*(__int64 **)(v6 + 40));
    v10 = v9;
    if ( v8 == sub_2FE4D20 )
    {
      v11 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v7 + 32LL);
      if ( v11 == sub_2D42F30 )
      {
        v12 = sub_AE2980(v10, 0)[1];
        v13 = 2;
        if ( v12 != 1 )
        {
          v13 = 3;
          if ( v12 != 2 )
          {
            v13 = 4;
            if ( v12 != 4 )
            {
              v13 = 5;
              if ( v12 != 8 )
              {
                v13 = 6;
                if ( v12 != 16 )
                {
                  v13 = 7;
                  if ( v12 != 32 )
                  {
                    v13 = 8;
                    if ( v12 != 64 )
                      v13 = 9 * (v12 == 128);
                  }
                }
              }
            }
          }
        }
      }
      else
      {
        v13 = v11(v7, v10, 0);
      }
    }
    else
    {
      v13 = ((__int64 (__fastcall *)(__int64, __int64))v8)(v7, v9);
    }
    v14 = *(_QWORD *)(a2 + 80);
    v15 = v13;
    v24 = v14;
    if ( v14 )
      sub_B96E90((__int64)&v24, v14, 1);
    v25 = *(_DWORD *)(a2 + 72);
    v16 = sub_33FB310(
            v6,
            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL),
            (__int64)&v24,
            v15,
            0,
            a4);
    v18 = v17;
    if ( v24 )
      sub_B91220((__int64)&v24, v24);
    *((_QWORD *)&v23 + 1) = v18;
    *(_QWORD *)&v23 = v16;
    return sub_33EC3B0(
             (_QWORD *)a1[1],
             (__int64 *)a2,
             **(_QWORD **)(a2 + 40),
             *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
             *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
             *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
             v23);
  }
}
