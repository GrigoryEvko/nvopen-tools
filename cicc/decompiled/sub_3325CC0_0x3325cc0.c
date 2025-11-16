// Function: sub_3325CC0
// Address: 0x3325cc0
//
__int64 __fastcall sub_3325CC0(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // r9
  __int64 v5; // r15
  unsigned __int16 *v6; // rax
  int v7; // r14d
  __int64 v8; // rax
  int v9; // edx
  unsigned __int16 v10; // ax
  __int64 v11; // rax
  __int64 v12; // r10
  __int64 v13; // rdx
  __int16 v14; // r12
  __int64 v15; // r14
  __int64 v17; // [rsp+20h] [rbp-A0h]
  __int64 v18; // [rsp+20h] [rbp-A0h]
  __int64 v19; // [rsp+28h] [rbp-98h]
  __int64 (__fastcall *v20)(__int64, __int64, unsigned int); // [rsp+30h] [rbp-90h]
  int v21; // [rsp+30h] [rbp-90h]
  __int64 v22; // [rsp+38h] [rbp-88h]
  __int64 v23; // [rsp+40h] [rbp-80h] BYREF
  int v24; // [rsp+48h] [rbp-78h]
  __int128 v25; // [rsp+50h] [rbp-70h] BYREF
  __int64 v26; // [rsp+60h] [rbp-60h]
  _QWORD v27[10]; // [rsp+70h] [rbp-50h] BYREF

  v3 = *(_QWORD *)(a2 + 80);
  v23 = v3;
  if ( v3 )
    sub_B96E90((__int64)&v23, v3, 1);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_QWORD *)(a1 + 16);
  v24 = *(_DWORD *)(a2 + 72);
  v6 = *(unsigned __int16 **)(a2 + 48);
  v7 = *v6;
  v17 = v4;
  v22 = *((_QWORD *)v6 + 1);
  v20 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v4 + 32LL);
  v8 = sub_2E79000(*(__int64 **)(v5 + 40));
  if ( v20 == sub_2D42F30 )
  {
    v9 = sub_AE2980(v8, 0)[1];
    v10 = 2;
    if ( v9 != 1 )
    {
      v10 = 3;
      if ( v9 != 2 )
      {
        v10 = 4;
        if ( v9 != 4 )
        {
          v10 = 5;
          if ( v9 != 8 )
          {
            v10 = 6;
            if ( v9 != 16 )
            {
              v10 = 7;
              if ( v9 != 32 )
              {
                v10 = 8;
                if ( v9 != 64 )
                  v10 = 9 * (v9 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v10 = v20(v17, v8, 0);
  }
  v11 = sub_33EE5B0(v5, *(_QWORD *)(a2 + 96), v10, 0, 0, 0, 0, 0);
  v12 = *(_QWORD *)(a1 + 16);
  memset(v27, 0, 32);
  v18 = v11;
  v19 = v13;
  LOBYTE(v11) = *(_BYTE *)(v11 + 108);
  v21 = v12;
  BYTE1(v11) = 1;
  v14 = v11;
  sub_2EAC2B0((__int64)&v25, *(_QWORD *)(v12 + 40));
  v15 = sub_33F1F00(
          v21,
          v7,
          v22,
          (unsigned int)&v23,
          (unsigned int)*(_QWORD *)(a1 + 16) + 288,
          0,
          v18,
          v19,
          v25,
          v26,
          v14,
          0,
          (__int64)v27,
          0);
  if ( v23 )
    sub_B91220((__int64)&v23, v23);
  return v15;
}
