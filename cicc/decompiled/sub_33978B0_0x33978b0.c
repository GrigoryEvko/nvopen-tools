// Function: sub_33978B0
// Address: 0x33978b0
//
_QWORD *__fastcall sub_33978B0(__int64 a1, __int64 a2)
{
  __int64 v4; // r15
  __int64 *v5; // rdx
  __int64 v6; // r13
  int v7; // edx
  int v8; // r14d
  __int64 v9; // rax
  int v10; // ecx
  int v11; // edx
  int v12; // r15d
  __int64 v13; // rax
  unsigned int v14; // r8d
  __int64 v15; // rax
  unsigned int v16; // r8d
  __int64 v17; // rdi
  unsigned int v18; // r9d
  __int64 (*v19)(); // rax
  int v20; // edx
  __int64 v21; // rax
  __int64 v22; // r11
  __int64 v23; // rsi
  int v24; // edx
  _QWORD *result; // rax
  char v26; // al
  int v27; // [rsp+8h] [rbp-88h]
  unsigned int v28; // [rsp+14h] [rbp-7Ch]
  unsigned int v29; // [rsp+18h] [rbp-78h]
  int v30; // [rsp+18h] [rbp-78h]
  __int64 v31; // [rsp+20h] [rbp-70h]
  int v32; // [rsp+20h] [rbp-70h]
  unsigned int v33; // [rsp+20h] [rbp-70h]
  __int64 *v34; // [rsp+28h] [rbp-68h]
  unsigned int v35; // [rsp+28h] [rbp-68h]
  __int64 v36; // [rsp+50h] [rbp-40h] BYREF
  int v37; // [rsp+58h] [rbp-38h]

  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 16LL);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v5 = *(__int64 **)(a2 - 8);
  else
    v5 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v31 = *v5;
  v6 = sub_338B750(a1, *v5);
  v8 = v7;
  v34 = *(__int64 **)(a2 + 8);
  v9 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL));
  v10 = sub_2D5BAE0(v4, v9, v34, 0);
  v12 = v11;
  v13 = *(_QWORD *)(v31 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v13 + 8) - 17 <= 1 )
    v13 = **(_QWORD **)(v13 + 16);
  v14 = *(_DWORD *)(v13 + 8);
  v15 = *(_QWORD *)(a2 + 8);
  v16 = v14 >> 8;
  if ( (unsigned int)*(unsigned __int8 *)(v15 + 8) - 17 <= 1 )
    v15 = **(_QWORD **)(v15 + 16);
  v17 = *(_QWORD *)(a1 + 856);
  v18 = *(_DWORD *)(v15 + 8) >> 8;
  v19 = *(__int64 (**)())(*(_QWORD *)v17 + 80LL);
  if ( v19 == sub_23CE2F0
    || (v30 = v10,
        v33 = v18,
        v35 = v16,
        v26 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD))v19)(v17, v16, v18),
        v16 = v35,
        v18 = v33,
        v10 = v30,
        !v26) )
  {
    v20 = *(_DWORD *)(a1 + 848);
    v21 = *(_QWORD *)a1;
    v36 = 0;
    v22 = *(_QWORD *)(a1 + 864);
    v37 = v20;
    if ( v21 )
    {
      if ( &v36 != (__int64 *)(v21 + 48) )
      {
        v23 = *(_QWORD *)(v21 + 48);
        v36 = v23;
        if ( v23 )
        {
          v27 = v10;
          v28 = v16;
          v29 = v18;
          v32 = v22;
          sub_B96E90((__int64)&v36, v23, 1);
          v10 = v27;
          v16 = v28;
          v18 = v29;
          LODWORD(v22) = v32;
        }
      }
    }
    v6 = sub_33F2D30(v22, (unsigned int)&v36, v10, v12, v6, v8, v16, v18);
    v8 = v24;
    if ( v36 )
      sub_B91220((__int64)&v36, v36);
  }
  v36 = a2;
  result = sub_337DC20(a1 + 8, &v36);
  *result = v6;
  *((_DWORD *)result + 2) = v8;
  return result;
}
