// Function: sub_3045D90
// Address: 0x3045d90
//
__int64 __fastcall sub_3045D90(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rdi
  __int64 (*v8)(void); // rax
  __int64 v9; // r15
  __int64 v10; // rsi
  __int64 *v11; // rdi
  __int64 v12; // rax
  int v13; // edx
  unsigned __int16 v14; // ax
  int v15; // r15d
  __int128 v16; // rax
  int v17; // r9d
  __int64 v18; // rax
  __int64 *v19; // rcx
  __int64 v20; // r8
  unsigned int v21; // edx
  unsigned int v22; // r9d
  __int64 v23; // rdx
  int v24; // eax
  __int64 v25; // rax
  __int64 v26; // r10
  __int64 v27; // r14
  unsigned __int16 *v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rsi
  __int64 v31; // r15
  __int64 v32; // rbx
  char v33; // al
  __int64 v34; // r14
  int v36; // [rsp+8h] [rbp-98h]
  unsigned int v37; // [rsp+10h] [rbp-90h]
  __int64 (__fastcall *v38)(__int64, __int64, unsigned int); // [rsp+18h] [rbp-88h]
  int v39; // [rsp+18h] [rbp-88h]
  __int64 v40; // [rsp+20h] [rbp-80h] BYREF
  int v41; // [rsp+28h] [rbp-78h]
  __int128 v42; // [rsp+30h] [rbp-70h]
  __int64 v43; // [rsp+40h] [rbp-60h]
  _QWORD v44[10]; // [rsp+50h] [rbp-50h] BYREF

  v7 = *(_QWORD *)(a1 + 537016);
  v8 = *(__int64 (**)(void))(*(_QWORD *)v7 + 144LL);
  if ( (char *)v8 == (char *)sub_3020010 )
    v9 = v7 + 960;
  else
    v9 = v8();
  v10 = *(_QWORD *)(a2 + 80);
  v40 = v10;
  if ( v10 )
    sub_B96E90((__int64)&v40, v10, 1);
  v11 = *(__int64 **)(a4 + 40);
  v41 = *(_DWORD *)(a2 + 72);
  v38 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v9 + 32LL);
  v12 = sub_2E79000(v11);
  if ( v38 == sub_2D42F30 )
  {
    v13 = sub_AE2980(v12, 0)[1];
    v14 = 2;
    if ( v13 != 1 )
    {
      v14 = 3;
      if ( v13 != 2 )
      {
        v14 = 4;
        if ( v13 != 4 )
        {
          v14 = 5;
          if ( v13 != 8 )
          {
            v14 = 6;
            if ( v13 != 16 )
            {
              v14 = 7;
              if ( v13 != 32 )
              {
                v14 = 8;
                if ( v13 != 64 )
                  v14 = 9 * (v13 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v14 = v38(v9, v12, 0);
  }
  v15 = v14;
  *(_QWORD *)&v16 = sub_3045D50(a1, a4, -1, v14, 0);
  v18 = sub_33FAF80(a4, 501, (unsigned int)&v40, v15, 0, v17, v16);
  v19 = *(__int64 **)(a2 + 40);
  memset(v44, 0, 32);
  v20 = v18;
  v22 = v21;
  v23 = *(_QWORD *)(v19[10] + 96);
  v24 = 0;
  if ( v23 )
  {
    v25 = *(_QWORD *)(v23 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v25 + 8) - 17 <= 1 )
      v25 = **(_QWORD **)(v25 + 16);
    v24 = *(_DWORD *)(v25 + 8) >> 8;
  }
  v26 = *v19;
  v27 = v19[5];
  LODWORD(v43) = v24;
  v28 = (unsigned __int16 *)(*(_QWORD *)(v20 + 48) + 16LL * v22);
  *(_QWORD *)&v42 = v23 & 0xFFFFFFFFFFFFFFFBLL;
  v29 = *((_QWORD *)v28 + 1);
  v30 = *v28;
  BYTE4(v43) = 0;
  v31 = v19[6];
  v32 = v19[1];
  v36 = v26;
  v37 = v22;
  v39 = v20;
  *((_QWORD *)&v42 + 1) = 0;
  v33 = sub_33CC4A0(a4, v30, v29);
  v34 = sub_33F4560(a4, v36, v32, (unsigned int)&v40, v39, v37, v27, v31, v42, v43, v33, 0, (__int64)v44);
  if ( v40 )
    sub_B91220((__int64)&v40, v40);
  return v34;
}
