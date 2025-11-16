// Function: sub_36E54E0
// Address: 0x36e54e0
//
void __fastcall sub_36E54E0(__int64 *a1, __int64 a2)
{
  __int64 v4; // rdi
  __int64 (*v5)(void); // rax
  __int64 v6; // r13
  __int64 v7; // rax
  _DWORD *v8; // rdx
  unsigned int v9; // eax
  __int64 v10; // r14
  __int64 (__fastcall *v11)(__int64, __int64, unsigned int); // r15
  __int64 v12; // rax
  int v13; // edx
  unsigned __int16 v14; // ax
  __int64 v15; // rax
  __int64 v16; // rsi
  _QWORD *v17; // r10
  __int64 v18; // r14
  __int64 v19; // rdx
  __int64 v20; // r15
  int v21; // r13d
  __int64 v22; // r8
  unsigned int v23; // ecx
  __int64 v24; // r13
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int128 v28; // [rsp-10h] [rbp-70h]
  unsigned int v29; // [rsp+0h] [rbp-60h]
  __int64 v30; // [rsp+8h] [rbp-58h]
  _QWORD *v31; // [rsp+10h] [rbp-50h]
  __int64 v32; // [rsp+18h] [rbp-48h]
  __int64 v33; // [rsp+20h] [rbp-40h] BYREF
  int v34; // [rsp+28h] [rbp-38h]

  v4 = a1[142];
  v5 = *(__int64 (**)(void))(*(_QWORD *)v4 + 144LL);
  if ( (char *)v5 == (char *)sub_3020010 )
    v6 = v4 + 960;
  else
    v6 = v5();
  v7 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 96LL);
  v8 = *(_DWORD **)(v7 + 24);
  v9 = *(_DWORD *)(v7 + 32);
  if ( v9 > 0x40 )
  {
    LODWORD(v10) = *v8;
  }
  else
  {
    LODWORD(v10) = 0;
    if ( v9 )
      v10 = (__int64)((_QWORD)v8 << (64 - (unsigned __int8)v9)) >> (64 - (unsigned __int8)v9);
  }
  v32 = a1[101];
  v11 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v32 + 32LL);
  v12 = sub_2E79000(*(__int64 **)(a1[8] + 40));
  if ( v11 == sub_2D42F30 )
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
    v14 = v11(v32, v12, 0);
  }
  v15 = sub_3045D50(v6, a1[8], v10, v14, 0);
  v16 = *(_QWORD *)(a2 + 80);
  v17 = (_QWORD *)a1[8];
  v18 = v15;
  v20 = v19;
  v21 = 2909 - ((*(_BYTE *)(a1[119] + 1264) == 0) - 1);
  v22 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  v23 = **(unsigned __int16 **)(a2 + 48);
  v33 = v16;
  if ( v16 )
  {
    v29 = v23;
    v30 = v22;
    v31 = v17;
    sub_B96E90((__int64)&v33, v16, 1);
    v23 = v29;
    v22 = v30;
    v17 = v31;
  }
  *((_QWORD *)&v28 + 1) = v20;
  *(_QWORD *)&v28 = v18;
  v34 = *(_DWORD *)(a2 + 72);
  v24 = sub_33F7740(v17, v21, (__int64)&v33, v23, v22, (__int64)&v33, v28);
  sub_34158F0(a1[8], a2, v24, v25, v26, v27);
  sub_3421DB0(v24);
  sub_33ECEA0((const __m128i *)a1[8], a2);
  if ( v33 )
    sub_B91220((__int64)&v33, v33);
}
