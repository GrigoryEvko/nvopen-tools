// Function: sub_3038030
// Address: 0x3038030
//
__int64 __fastcall sub_3038030(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rcx
  __int64 v7; // rsi
  __int64 v8; // rax
  _DWORD *v9; // rax
  unsigned __int16 v10; // bx
  __int64 v11; // rcx
  int v12; // eax
  unsigned int v13; // edx
  int v14; // r9d
  __int64 v15; // r12
  unsigned __int16 v17; // ax
  __int128 v18; // [rsp-20h] [rbp-90h]
  __int64 (__fastcall *v19)(__int64, __int64, unsigned int); // [rsp+8h] [rbp-68h]
  __int64 v20; // [rsp+10h] [rbp-60h]
  __int64 v21; // [rsp+18h] [rbp-58h]
  unsigned int v22; // [rsp+18h] [rbp-58h]
  __int64 v23; // [rsp+20h] [rbp-50h]
  __int64 v24; // [rsp+30h] [rbp-40h] BYREF
  int v25; // [rsp+38h] [rbp-38h]

  v5 = a2;
  v7 = *(_QWORD *)(a2 + 80);
  v24 = v7;
  if ( v7 )
  {
    v21 = v5;
    sub_B96E90((__int64)&v24, v7, 1);
    v5 = v21;
  }
  v20 = v5;
  v25 = *(_DWORD *)(v5 + 72);
  v19 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)a1 + 32LL);
  v22 = sub_33D0880(v5);
  v8 = sub_2E79000(*(__int64 **)(a4 + 40));
  if ( v19 == sub_2D42F30 )
  {
    v9 = sub_AE2980(v8, v22);
    v10 = 2;
    v11 = v20;
    v12 = v9[1];
    if ( v12 != 1 )
    {
      v10 = 3;
      if ( v12 != 2 )
      {
        v10 = 4;
        if ( v12 != 4 )
        {
          v10 = 5;
          if ( v12 != 8 )
          {
            v10 = 6;
            if ( v12 != 16 )
            {
              v10 = 7;
              if ( v12 != 32 )
              {
                v10 = 8;
                if ( v12 != 64 )
                  v10 = 9 * (v12 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v17 = v19(a1, v8, v22);
    v11 = v20;
    v10 = v17;
  }
  v23 = sub_33ED290(a4, *(_QWORD *)(v11 + 96), (unsigned int)&v24, v10, 0, 0, 1, 0);
  *((_QWORD *)&v18 + 1) = v13 | a3 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v18 = v23;
  v15 = sub_33FAF80(a4, 501, (unsigned int)&v24, v10, 0, v14, v18);
  if ( v24 )
    sub_B91220((__int64)&v24, v24);
  return v15;
}
