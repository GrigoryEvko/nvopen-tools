// Function: sub_930A90
// Address: 0x930a90
//
__int64 __fastcall sub_930A90(__int64 a1, __int64 *a2, _QWORD *a3, char a4, char a5)
{
  __m128i *i; // r13
  __int64 v8; // r14
  _QWORD *v9; // rax
  unsigned __int64 v10; // r9
  unsigned __int64 v11; // rdx
  _QWORD *v12; // rax
  __int64 v13; // rax
  __int64 v15; // rax
  __int64 v16; // r14
  __int64 v17; // r15
  __int64 v18; // rax
  __int64 v19; // r11
  __int64 v20; // rdi
  __int64 (__fastcall *v21)(__int64, unsigned int, __m128i *, __int64); // rax
  __int64 v22; // rax
  __int64 v23; // r15
  int v24; // r8d
  __int64 v25; // r14
  __int64 v26; // rax
  unsigned __int8 v27; // al
  __int64 v28; // rax
  unsigned int *v29; // r14
  unsigned int *v30; // r15
  __int64 v31; // rdx
  __int64 v32; // rsi
  __int64 v33; // rax
  __int64 v34; // rdx
  unsigned int v35; // r13d
  __int64 v36; // rax
  __int64 v37; // r13
  unsigned int *v38; // r13
  __int64 v39; // rdx
  __int64 v40; // rsi
  __int64 v41; // rax
  int v42; // [rsp+0h] [rbp-F0h]
  int v43; // [rsp+4h] [rbp-ECh]
  __int64 v44; // [rsp+10h] [rbp-E0h]
  unsigned int *v45; // [rsp+10h] [rbp-E0h]
  __int64 v46; // [rsp+10h] [rbp-E0h]
  __int64 v47; // [rsp+18h] [rbp-D8h]
  _BYTE v48[32]; // [rsp+20h] [rbp-D0h] BYREF
  __int16 v49; // [rsp+40h] [rbp-B0h]
  _BYTE v50[32]; // [rsp+50h] [rbp-A0h] BYREF
  __int16 v51; // [rsp+70h] [rbp-80h]
  char v52[8]; // [rsp+80h] [rbp-70h] BYREF
  __m128i *v53; // [rsp+88h] [rbp-68h]

  if ( a4 == 1 && !a5 )
  {
    sub_926800((__int64)v52, a1, (__int64)a2);
    i = v53;
LABEL_4:
    v8 = a3[1];
    v9 = (_QWORD *)*a3;
    v10 = v8 + 1;
    if ( (_QWORD *)*a3 == a3 + 2 )
      v11 = 15;
    else
      v11 = a3[2];
    if ( v10 > v11 )
    {
      sub_2240BB0(a3, a3[1], 0, 0, 1);
      v9 = (_QWORD *)*a3;
      v10 = v8 + 1;
    }
    *((_BYTE *)v9 + v8) = 42;
    v12 = (_QWORD *)*a3;
    a3[1] = v10;
    *((_BYTE *)v12 + v8 + 1) = 0;
    goto LABEL_9;
  }
  if ( !sub_91B770(*a2) )
  {
    i = sub_92F410(a1, (__int64)a2);
    v13 = i->m128i_i64[1];
    if ( *(_BYTE *)(v13 + 8) != 14 )
      return (__int64)i;
    goto LABEL_31;
  }
  sub_926800((__int64)v52, a1, (__int64)a2);
  v15 = *a2;
  for ( i = v53; *(_BYTE *)(v15 + 140) == 12; v15 = *(_QWORD *)(v15 + 160) )
    ;
  v16 = 8LL * *(_QWORD *)(v15 + 128);
  if ( (unsigned __int64)(v16 - 1) > 0x3F || ((v16 - 1) & v16) != 0 )
    goto LABEL_4;
  v17 = v53->m128i_i64[1];
  v18 = sub_BCCE00(*(_QWORD *)(a1 + 40), (unsigned int)v16);
  v19 = sub_BCE760(v18, *(_DWORD *)(v17 + 8) >> 8);
  v49 = 257;
  v47 = a1 + 48;
  if ( v19 == i->m128i_i64[1] )
  {
    v23 = (__int64)i;
    goto LABEL_23;
  }
  v20 = *(_QWORD *)(a1 + 128);
  v21 = *(__int64 (__fastcall **)(__int64, unsigned int, __m128i *, __int64))(*(_QWORD *)v20 + 120LL);
  if ( (char *)v21 == (char *)sub_920130 )
  {
    if ( i->m128i_i8[0] > 0x15u )
    {
LABEL_34:
      v51 = 257;
      v23 = sub_B51D30(49, i, v19, v50, 0, 0);
      if ( (unsigned __int8)sub_920620(v23) )
      {
        v34 = *(_QWORD *)(a1 + 144);
        v35 = *(_DWORD *)(a1 + 152);
        if ( v34 )
          sub_B99FD0(v23, 3, v34);
        sub_B45150(v23, v35);
      }
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
        *(_QWORD *)(a1 + 136),
        v23,
        v48,
        *(_QWORD *)(v47 + 56),
        *(_QWORD *)(v47 + 64));
      v36 = *(_QWORD *)(a1 + 48);
      v37 = 16LL * *(unsigned int *)(a1 + 56);
      v45 = (unsigned int *)(v36 + v37);
      if ( v36 != v36 + v37 )
      {
        v38 = *(unsigned int **)(a1 + 48);
        do
        {
          v39 = *((_QWORD *)v38 + 1);
          v40 = *v38;
          v38 += 4;
          sub_B99FD0(v23, v40, v39);
        }
        while ( v45 != v38 );
      }
      goto LABEL_23;
    }
    v44 = v19;
    if ( (unsigned __int8)sub_AC4810(49) )
      v22 = sub_ADAB70(49, i, v44, 0);
    else
      v22 = sub_AA93C0(49, i, v44);
    v19 = v44;
    v23 = v22;
  }
  else
  {
    v46 = v19;
    v41 = v21(v20, 49u, i, v19);
    v19 = v46;
    v23 = v41;
  }
  if ( !v23 )
    goto LABEL_34;
LABEL_23:
  v49 = 257;
  v24 = unk_4D0463C;
  if ( unk_4D0463C )
    v24 = sub_90AA40(*(_QWORD *)(a1 + 32), v23);
  v42 = v24;
  v25 = sub_BCCE00(*(_QWORD *)(a1 + 40), (unsigned int)v16);
  v26 = sub_AA4E30(*(_QWORD *)(a1 + 96));
  v27 = sub_AE5020(v26, v25);
  v51 = 257;
  v43 = v27;
  v28 = sub_BD2C40(80, unk_3F10A14);
  i = (__m128i *)v28;
  if ( v28 )
    sub_B4D190(v28, v25, v23, (unsigned int)v50, v42, v43, 0, 0);
  (*(void (__fastcall **)(_QWORD, __m128i *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
    *(_QWORD *)(a1 + 136),
    i,
    v48,
    *(_QWORD *)(v47 + 56),
    *(_QWORD *)(v47 + 64));
  v29 = *(unsigned int **)(a1 + 48);
  v30 = &v29[4 * *(unsigned int *)(a1 + 56)];
  while ( v30 != v29 )
  {
    v31 = *((_QWORD *)v29 + 1);
    v32 = *v29;
    v29 += 4;
    sub_B99FD0(i, v32, v31);
  }
LABEL_9:
  v13 = i->m128i_i64[1];
  if ( *(_BYTE *)(v13 + 8) != 14 )
    return (__int64)i;
LABEL_31:
  if ( !(*(_DWORD *)(v13 + 8) >> 8) )
    return (__int64)i;
  v33 = sub_BCE3C0(*(_QWORD *)v13, 0);
  return sub_92CA20(a1, (__int64)i, v33);
}
