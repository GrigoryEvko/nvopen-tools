// Function: sub_35D4FB0
// Address: 0x35d4fb0
//
__int64 __fastcall sub_35D4FB0(__int64 a1, unsigned __int8 **a2)
{
  __int64 (*v3)(); // rax
  __int64 v4; // r8
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // rdi
  __int64 (__fastcall *v10)(__int64, unsigned __int16); // r13
  __int64 (__fastcall *v11)(__int64, __int64, unsigned int); // rax
  __int64 v12; // rsi
  int v13; // eax
  __int64 v14; // r9
  __int64 *v15; // rax
  __int64 *v16; // r14
  __int64 *v17; // rax
  __int64 v18; // rax
  __int32 v19; // r12d
  unsigned __int8 *v20; // rsi
  __int64 v21; // r13
  unsigned __int8 v22; // [rsp+17h] [rbp-89h]
  __int64 v23; // [rsp+18h] [rbp-88h]
  __int64 *v24; // [rsp+20h] [rbp-80h]
  __int64 v26; // [rsp+38h] [rbp-68h]
  unsigned __int8 *v27; // [rsp+48h] [rbp-58h] BYREF
  __int64 v28[10]; // [rsp+50h] [rbp-50h] BYREF

  v3 = *(__int64 (**)())(**(_QWORD **)(a1 + 16) + 2216LL);
  if ( v3 == sub_302E1B0 )
    goto LABEL_2;
  v22 = v3();
  if ( !v22 || !*(_DWORD *)(a1 + 144) )
    goto LABEL_2;
  v6 = *(_QWORD *)(*(_QWORD *)a1 + 328LL);
  v7 = sub_2E79000(*(__int64 **)a1);
  v8 = *(_QWORD *)(a1 + 16);
  v9 = v7;
  v10 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v8 + 552LL);
  v11 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v8 + 32LL);
  if ( v11 == sub_2D42F30 )
  {
    v12 = 2;
    v13 = sub_AE2980(v9, 0)[1];
    if ( v13 != 1 )
    {
      v12 = 3;
      if ( v13 != 2 )
      {
        v12 = 4;
        if ( v13 != 4 )
        {
          v12 = 5;
          if ( v13 != 8 )
          {
            v12 = 6;
            if ( v13 != 16 )
            {
              v12 = 7;
              if ( v13 != 32 )
              {
                v12 = 8;
                if ( v13 != 64 )
                  v12 = 9 * (unsigned int)(v13 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v12 = (unsigned int)v11(*(_QWORD *)(a1 + 16), v9, 0);
  }
  v23 = v10 == sub_2EC09E0
      ? *(_QWORD *)(v8 + 8LL * (unsigned __int16)v12 + 112)
      : ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v10)(v8, v12, 0);
  v15 = *(__int64 **)(a1 + 136);
  v24 = &v15[*(unsigned int *)(a1 + 144)];
  if ( v24 == v15 )
  {
LABEL_2:
    LODWORD(v4) = 0;
  }
  else
  {
    v16 = *(__int64 **)(a1 + 136);
    v4 = 0;
    do
    {
      v26 = *v16;
      v18 = *(_QWORD *)(a1 + 128);
      if ( v18 != *v16 || !v18 )
      {
        v19 = sub_2EC06C0(*(_QWORD *)(*(_QWORD *)a1 + 32LL), v23, byte_3F871B3, 0, v4, v14);
        v20 = *a2;
        v21 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL) - 400LL;
        v27 = v20;
        if ( v20 )
        {
          sub_B96E90((__int64)&v27, (__int64)v20, 1);
          v28[0] = (__int64)v27;
          if ( v27 )
          {
            sub_B976B0((__int64)&v27, v27, (__int64)v28);
            v27 = 0;
          }
        }
        else
        {
          v28[0] = 0;
        }
        v28[1] = 0;
        v28[2] = 0;
        v17 = (__int64 *)sub_2E311E0(v6);
        sub_2F26260(v6, v17, v28, v21, v19);
        if ( v28[0] )
          sub_B91220((__int64)v28, v28[0]);
        if ( v27 )
          sub_B91220((__int64)&v27, (__int64)v27);
        sub_35D4CD0(a1, v6, v26, v19);
        v4 = v22;
      }
      ++v16;
    }
    while ( v24 != v16 );
  }
  return (unsigned int)v4;
}
