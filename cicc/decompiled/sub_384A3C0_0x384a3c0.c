// Function: sub_384A3C0
// Address: 0x384a3c0
//
void __fastcall sub_384A3C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5)
{
  __int64 v8; // rsi
  unsigned __int64 *v9; // rax
  unsigned __int64 v10; // r15
  __int64 v11; // rsi
  __int64 v12; // rax
  __int16 v13; // dx
  __int64 v14; // rax
  int v15; // r9d
  unsigned __int16 *v16; // rax
  unsigned __int8 *v17; // rax
  int v18; // edx
  __int64 v19; // rdx
  unsigned __int16 *v20; // rax
  int v21; // r9d
  unsigned __int8 *v22; // rax
  __int64 v23; // rsi
  int v24; // edx
  int v25; // [rsp+28h] [rbp-78h]
  __int64 v26; // [rsp+30h] [rbp-70h] BYREF
  __int64 v27; // [rsp+38h] [rbp-68h]
  __int64 v28; // [rsp+40h] [rbp-60h] BYREF
  __int64 v29; // [rsp+48h] [rbp-58h]
  __int64 v30; // [rsp+50h] [rbp-50h] BYREF
  int v31; // [rsp+58h] [rbp-48h]
  __int16 v32; // [rsp+60h] [rbp-40h] BYREF
  __int64 v33; // [rsp+68h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 80);
  v26 = 0;
  LODWORD(v27) = 0;
  v28 = 0;
  LODWORD(v29) = 0;
  v30 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v30, v8, 1);
  v31 = *(_DWORD *)(a2 + 72);
  v9 = *(unsigned __int64 **)(a2 + 40);
  v10 = *v9;
  v11 = v9[1];
  v12 = *(_QWORD *)(*v9 + 48) + 16LL * *((unsigned int *)v9 + 2);
  v13 = *(_WORD *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  v32 = v13;
  v33 = v14;
  if ( v13 )
  {
    if ( (unsigned __int16)(v13 - 17) > 0xD3u )
    {
      if ( (unsigned __int16)(v13 - 2) <= 7u || (unsigned __int16)(v13 - 176) <= 0x1Fu )
        goto LABEL_6;
LABEL_9:
      sub_375E6F0(a1, v10, v11, (__int64)&v26, (__int64)&v28);
      goto LABEL_10;
    }
  }
  else if ( !sub_30070B0((__int64)&v32) )
  {
    if ( sub_3007070((__int64)&v32) )
    {
LABEL_6:
      sub_375E510(a1, v10, v11, (__int64)&v26, (__int64)&v28);
      goto LABEL_10;
    }
    goto LABEL_9;
  }
  sub_375E8D0(a1, v10, v11, (__int64)&v26, (__int64)&v28);
LABEL_10:
  v16 = (unsigned __int16 *)(*(_QWORD *)(v26 + 48) + 16LL * (unsigned int)v27);
  v17 = sub_33FAF80(*(_QWORD *)(a1 + 8), 52, (__int64)&v30, *v16, *((_QWORD *)v16 + 1), v15, a5);
  v25 = v18;
  v19 = v28;
  *(_QWORD *)a3 = v17;
  *(_DWORD *)(a3 + 8) = v25;
  v20 = (unsigned __int16 *)(*(_QWORD *)(v19 + 48) + 16LL * (unsigned int)v29);
  v22 = sub_33FAF80(*(_QWORD *)(a1 + 8), 52, (__int64)&v30, *v20, *((_QWORD *)v20 + 1), v21, a5);
  v23 = v30;
  *(_QWORD *)a4 = v22;
  *(_DWORD *)(a4 + 8) = v24;
  if ( v23 )
    sub_B91220((__int64)&v30, v23);
}
