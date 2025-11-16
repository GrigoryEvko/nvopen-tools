// Function: sub_38036B0
// Address: 0x38036b0
//
unsigned __int8 *__fastcall sub_38036B0(__int64 *a1, __int64 a2, __m128i a3)
{
  __int16 *v5; // rax
  __int64 v6; // r10
  unsigned __int16 v7; // si
  __int64 v8; // r8
  __int64 v9; // r12
  __int64 (__fastcall *v10)(__int64, __int64, unsigned int, __int64); // rax
  unsigned int v11; // r13d
  __int64 v12; // rax
  unsigned __int16 v13; // dx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdx
  char v17; // al
  unsigned int v18; // eax
  __int64 v19; // rdi
  __int16 v20; // ax
  __int64 v21; // rdx
  int v22; // edx
  __int64 v23; // rsi
  __int64 v24; // r14
  int v25; // r9d
  __int64 v26; // rsi
  unsigned __int8 *v27; // r14
  __int64 v29; // rax
  int v30; // eax
  __int64 v31; // rdx
  int v32; // [rsp+8h] [rbp-88h]
  __int64 v33; // [rsp+10h] [rbp-80h]
  unsigned __int16 v34; // [rsp+1Ch] [rbp-74h]
  __int16 v35; // [rsp+1Eh] [rbp-72h]
  unsigned __int16 v36; // [rsp+20h] [rbp-70h] BYREF
  __int64 v37; // [rsp+28h] [rbp-68h]
  __int64 v38; // [rsp+30h] [rbp-60h] BYREF
  int v39; // [rsp+38h] [rbp-58h]
  __int64 v40; // [rsp+40h] [rbp-50h] BYREF
  __int64 v41; // [rsp+48h] [rbp-48h]
  __int64 v42; // [rsp+50h] [rbp-40h]

  v5 = *(__int16 **)(a2 + 48);
  v6 = *a1;
  v7 = *v5;
  v8 = *((_QWORD *)v5 + 1);
  v9 = *(_QWORD *)(a1[1] + 64);
  v34 = *v5;
  v10 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v10 == sub_2D56A50 )
  {
    HIWORD(v11) = 0;
    sub_2FE6CC0((__int64)&v40, v6, v9, v7, v8);
    v35 = v41;
    v33 = v42;
  }
  else
  {
    v30 = v10(v6, v9, v34, v8);
    v33 = v31;
    HIWORD(v11) = HIWORD(v30);
    v35 = v30;
  }
  v12 = *(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL);
  v13 = *(_WORD *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  v36 = v13;
  v37 = v14;
  if ( v13 )
  {
    if ( v13 == 1 || (unsigned __int16)(v13 - 504) <= 7u )
      BUG();
    v29 = 16LL * (v13 - 1);
    v16 = *(_QWORD *)&byte_444C4A0[v29];
    v17 = byte_444C4A0[v29 + 8];
  }
  else
  {
    v40 = sub_3007260((__int64)&v36);
    v41 = v15;
    v16 = v40;
    v17 = v41;
  }
  v38 = v16;
  LOBYTE(v39) = v17;
  v18 = sub_CA1930(&v38);
  v19 = a1[1];
  switch ( v18 )
  {
    case 1u:
      v20 = 2;
      break;
    case 2u:
      v20 = 3;
      break;
    case 4u:
      v20 = 4;
      break;
    case 8u:
      v20 = 5;
      break;
    case 0x10u:
      v20 = 6;
      break;
    case 0x20u:
      v20 = 7;
      break;
    case 0x40u:
      v20 = 8;
      break;
    case 0x80u:
      v20 = 9;
      break;
    default:
      v20 = sub_3007020(*(_QWORD **)(v19 + 64), v18);
      v19 = a1[1];
      goto LABEL_16;
  }
  v21 = 0;
LABEL_16:
  sub_33FB890(v19, v20, v21, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), a3);
  v23 = *(_QWORD *)(a2 + 80);
  v24 = a1[1];
  v25 = v22;
  v38 = v23;
  if ( v23 )
  {
    v32 = v22;
    sub_B96E90((__int64)&v38, v23, 1);
    v25 = v32;
  }
  v39 = *(_DWORD *)(a2 + 72);
  if ( v34 == 11 )
  {
    v26 = 236;
  }
  else if ( v35 == 11 )
  {
    v26 = 237;
  }
  else if ( v34 == 10 )
  {
    v26 = 240;
  }
  else
  {
    if ( v35 != 10 )
      sub_C64ED0("Attempt at an invalid promotion-related conversion", 1u);
    v26 = 241;
  }
  LOWORD(v11) = v35;
  v27 = sub_33FAF80(v24, v26, (__int64)&v38, v11, v33, v25, a3);
  if ( v38 )
    sub_B91220((__int64)&v38, v38);
  return v27;
}
