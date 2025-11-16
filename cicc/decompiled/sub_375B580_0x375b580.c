// Function: sub_375B580
// Address: 0x375b580
//
unsigned __int8 *__fastcall sub_375B580(__int64 a1, __int64 a2, __m128i a3, __int64 a4, __int64 a5, __int64 a6)
{
  _DWORD *v9; // r15
  __int64 v10; // r12
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 *v13; // rdi
  __int64 (__fastcall *v14)(_DWORD *, __int64, __int64, _QWORD, __int64); // r13
  __int64 v15; // rax
  __int64 v16; // rax
  int v17; // r9d
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // r13
  unsigned __int16 v21; // si
  unsigned int v22; // eax
  char v23; // bl
  bool v24; // al
  unsigned __int8 *v25; // r12
  __int64 v27; // [rsp+8h] [rbp-68h]
  __int64 v28; // [rsp+8h] [rbp-68h]
  __int64 v29; // [rsp+20h] [rbp-50h] BYREF
  int v30; // [rsp+28h] [rbp-48h]
  _QWORD v31[8]; // [rsp+30h] [rbp-40h] BYREF

  v9 = *(_DWORD **)a1;
  v10 = *(_QWORD *)(a1 + 8);
  v11 = *(_QWORD *)(a2 + 80);
  v29 = v11;
  if ( v11 )
    sub_B96E90((__int64)&v29, v11, 1);
  v12 = *(_QWORD *)(v10 + 64);
  v13 = *(__int64 **)(v10 + 40);
  v30 = *(_DWORD *)(a2 + 72);
  v27 = v12;
  v14 = *(__int64 (__fastcall **)(_DWORD *, __int64, __int64, _QWORD, __int64))(*(_QWORD *)v9 + 528LL);
  v15 = sub_2E79000(v13);
  v16 = v14(v9, v15, v27, (unsigned int)a5, a6);
  v31[0] = a5;
  v18 = v16;
  v31[1] = a6;
  v20 = v19;
  if ( (_WORD)a5 )
  {
    v21 = a5 - 17;
    if ( (unsigned __int16)(a5 - 10) > 6u && (unsigned __int16)(a5 - 126) > 0x31u )
    {
      if ( v21 <= 0xD3u )
      {
LABEL_7:
        v22 = v9[17];
        goto LABEL_11;
      }
LABEL_10:
      v22 = v9[15];
      goto LABEL_11;
    }
    if ( v21 <= 0xD3u )
      goto LABEL_7;
  }
  else
  {
    v28 = v16;
    v23 = sub_3007030((__int64)v31);
    v24 = sub_30070B0((__int64)v31);
    v18 = v28;
    if ( v24 )
      goto LABEL_7;
    if ( !v23 )
      goto LABEL_10;
  }
  v22 = v9[16];
LABEL_11:
  if ( v22 > 2 )
    BUG();
  v25 = sub_33FAF80(v10, 215 - v22, (__int64)&v29, v18, v20, v17, a3);
  if ( v29 )
    sub_B91220((__int64)&v29, v29);
  return v25;
}
