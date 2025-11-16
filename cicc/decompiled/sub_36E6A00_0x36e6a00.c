// Function: sub_36E6A00
// Address: 0x36e6a00
//
void __fastcall sub_36E6A00(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v5; // rsi
  unsigned __int16 v6; // ax
  int v7; // r15d
  __int64 v8; // rdi
  __int64 v9; // rax
  _QWORD *v10; // rsi
  __int64 v11; // rax
  int v12; // edx
  __int64 v13; // rsi
  unsigned __int64 v14; // rdx
  __int64 v15; // rsi
  unsigned __int64 v16; // rax
  __int64 v17; // rdi
  unsigned int v18; // eax
  __int64 v19; // r9
  int v20; // edx
  _QWORD *v21; // rdi
  unsigned __int64 v22; // rcx
  __int64 v23; // r8
  const __m128i *v24; // rax
  __int64 v25; // r13
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  unsigned __int8 *v29; // [rsp+0h] [rbp-E0h]
  int v30; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v31; // [rsp+18h] [rbp-C8h]
  int v32; // [rsp+20h] [rbp-C0h]
  int v33; // [rsp+24h] [rbp-BCh]
  __int64 v34; // [rsp+28h] [rbp-B8h]
  __int64 v35; // [rsp+30h] [rbp-B0h] BYREF
  int v36; // [rsp+38h] [rbp-A8h]
  __int64 v37; // [rsp+40h] [rbp-A0h] BYREF
  int v38; // [rsp+48h] [rbp-98h]
  unsigned __int8 *v39; // [rsp+50h] [rbp-90h] BYREF
  int v40; // [rsp+58h] [rbp-88h]
  unsigned __int8 *v41; // [rsp+60h] [rbp-80h]
  int v42; // [rsp+68h] [rbp-78h]
  __int64 v43; // [rsp+70h] [rbp-70h]
  int v44; // [rsp+78h] [rbp-68h]
  unsigned __int64 v45; // [rsp+80h] [rbp-60h]
  int v46; // [rsp+88h] [rbp-58h]
  __m128i v47; // [rsp+90h] [rbp-50h]
  __m128i v48; // [rsp+A0h] [rbp-40h]

  v5 = *(_QWORD *)(a2 + 80);
  v35 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v35, v5, 1);
  v36 = *(_DWORD *)(a2 + 72);
  v6 = *(_WORD *)(a2 + 96);
  if ( v6 == 64 )
  {
    v7 = 2885;
  }
  else if ( v6 > 0x40u )
  {
    switch ( v6 )
    {
      case 0x99u:
        v7 = 2877;
        break;
      case 0xA7u:
        v7 = 2879;
        break;
      case 0x95u:
        v7 = 2876;
        break;
      default:
        goto LABEL_30;
    }
  }
  else if ( v6 == 58 )
  {
    v7 = 2883;
  }
  else
  {
    if ( v6 <= 0x3Au )
    {
      if ( v6 == 7 )
      {
        v7 = 2882;
        goto LABEL_9;
      }
      if ( v6 == 13 )
      {
        v7 = 2878;
        goto LABEL_9;
      }
LABEL_30:
      BUG();
    }
    if ( v6 != 60 )
      goto LABEL_30;
    v7 = 2884;
  }
LABEL_9:
  v8 = *(_QWORD *)(a1 + 64);
  v9 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL) + 96LL);
  v10 = *(_QWORD **)(v9 + 24);
  if ( *(_DWORD *)(v9 + 32) > 0x40u )
    v10 = (_QWORD *)*v10;
  v29 = sub_3400BD0(v8, (__int64)v10, (__int64)&v35, 8, 0, 1u, a3, 0);
  v11 = *(_QWORD *)(a2 + 40);
  v30 = v12;
  v13 = *(_QWORD *)(v11 + 120);
  v14 = *(_QWORD *)(v11 + 128);
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  sub_36DF750(a1, v13, v14, (__int64)&v37, (__int64)&v39, a3);
  v15 = *(_QWORD *)(a2 + 80);
  v34 = v37;
  v37 = v15;
  v33 = v38;
  v16 = (unsigned __int64)v39;
  v39 = v29;
  v31 = v16;
  LODWORD(v16) = v40;
  v40 = v30;
  v32 = v16;
  if ( v15 )
    sub_B96E90((__int64)&v37, v15, 1);
  v17 = *(_QWORD *)(a2 + 112);
  v38 = *(_DWORD *)(a2 + 72);
  v18 = sub_36D7800(v17);
  v41 = sub_3400BD0(*(_QWORD *)(a1 + 64), v18, (__int64)&v37, 7, 0, 1u, a3, 0);
  v42 = v20;
  if ( v37 )
    sub_B91220((__int64)&v37, v37);
  v21 = *(_QWORD **)(a1 + 64);
  v22 = *(_QWORD *)(a2 + 48);
  v23 = *(unsigned int *)(a2 + 68);
  v43 = v34;
  v44 = v33;
  v45 = v31;
  v46 = v32;
  v24 = *(const __m128i **)(a2 + 40);
  v47 = _mm_loadu_si128(v24 + 10);
  v48 = _mm_loadu_si128(v24);
  v25 = sub_33E66D0(v21, v7, (__int64)&v35, v22, v23, v19, (unsigned __int64 *)&v39, 6);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v25, v26, v27, v28);
  sub_3421DB0(v25);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v35 )
    sub_B91220((__int64)&v35, v35);
}
