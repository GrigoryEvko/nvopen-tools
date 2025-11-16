// Function: sub_1E10FF0
// Address: 0x1e10ff0
//
__int64 __fastcall sub_1E10FF0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 (*v3)(); // rax
  __int64 (*v4)(); // rax
  char v5; // bl
  __int64 v6; // r15
  _QWORD *v7; // r12
  __int64 (*v8)(); // rax
  int v9; // r13d
  char v10; // r15
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 (*v15)(); // rax
  __int64 v16; // rsi
  __int64 v17; // r8
  __int64 v18; // rdi
  __int64 v19; // rbx
  __int64 v20; // rdi
  int v21; // eax
  __int64 v22; // r12
  __int64 (*v23)(void); // rax
  __int64 result; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rbx
  __int64 v30; // rdi
  __int64 (*v31)(); // rax
  unsigned int v32; // eax
  _OWORD *v33; // rax
  __int64 v34; // r15
  _QWORD *v35; // r13
  __int64 v36; // rbx
  unsigned int v37; // eax

  **(_QWORD **)(a1 + 352) |= 1uLL;
  **(_QWORD **)(a1 + 352) |= 4uLL;
  v2 = *(_QWORD *)(a1 + 16);
  v3 = *(__int64 (**)())(*(_QWORD *)v2 + 112LL);
  if ( v3 == sub_1D00B10 )
    goto LABEL_2;
  if ( !v3() )
  {
    v2 = *(_QWORD *)(a1 + 16);
LABEL_2:
    *(_QWORD *)(a1 + 40) = 0;
    goto LABEL_3;
  }
  v36 = sub_145CBF0((__int64 *)(a1 + 120), 384, 16);
  sub_1E6ACC0(v36, a1);
  *(_QWORD *)(a1 + 40) = v36;
  v2 = *(_QWORD *)(a1 + 16);
LABEL_3:
  *(_QWORD *)(a1 + 48) = 0;
  v4 = *(__int64 (**)())(*(_QWORD *)v2 + 48LL);
  if ( v4 == sub_1D90020 )
    BUG();
  v5 = *(_BYTE *)(v4() + 24);
  if ( v5 )
    v5 = !sub_15602E0((_QWORD *)(*(_QWORD *)a1 + 112LL), "no-realign-stack", 0x10u);
  v6 = *(_QWORD *)(a1 + 16);
  v7 = (_QWORD *)(*(_QWORD *)a1 + 112LL);
  if ( (unsigned __int8)sub_1560180((__int64)v7, 48) )
  {
    v9 = 0;
    if ( (unsigned __int8)sub_1560180((__int64)v7, 48) )
      v9 = sub_15603C0(v7, -1);
  }
  else
  {
    v8 = *(__int64 (**)())(*(_QWORD *)v6 + 48LL);
    if ( v8 == sub_1D90020 )
      BUG();
    v9 = *(_DWORD *)(((__int64 (__fastcall *)(__int64))v8)(v6) + 12);
  }
  v10 = 0;
  if ( v5 )
    v10 = sub_1560180(*(_QWORD *)a1 + 112LL, 48);
  v11 = sub_145CBF0((__int64 *)(a1 + 120), 680, 16);
  *(_WORD *)(v11 + 64) = 0;
  *(_QWORD *)(v11 + 120) = 0x2000000000LL;
  *(_DWORD *)v11 = v9;
  *(_BYTE *)(v11 + 4) = v5;
  *(_BYTE *)(v11 + 5) = v10;
  *(_QWORD *)(v11 + 8) = 0;
  *(_QWORD *)(v11 + 16) = 0;
  *(_QWORD *)(v11 + 24) = 0;
  *(_QWORD *)(v11 + 32) = 0;
  *(_BYTE *)(v11 + 40) = 0;
  *(_QWORD *)(v11 + 48) = 0;
  *(_QWORD *)(v11 + 56) = 0;
  *(_DWORD *)(v11 + 68) = -1;
  *(_QWORD *)(v11 + 72) = -1;
  *(_QWORD *)(v11 + 80) = 0;
  *(_QWORD *)(v11 + 88) = 0;
  *(_QWORD *)(v11 + 96) = 0;
  *(_BYTE *)(v11 + 104) = 0;
  *(_QWORD *)(v11 + 112) = v11 + 128;
  *(_QWORD *)(v11 + 640) = 0;
  *(_QWORD *)(v11 + 648) = 0;
  *(_WORD *)(v11 + 656) = 0;
  *(_QWORD *)(v11 + 664) = 0;
  *(_QWORD *)(v11 + 672) = 0;
  *(_QWORD *)(a1 + 56) = v11;
  if ( (unsigned __int8)sub_1560180(*(_QWORD *)a1 + 112LL, 48) )
  {
    v34 = *(_QWORD *)(a1 + 56);
    v35 = (_QWORD *)(*(_QWORD *)a1 + 112LL);
    if ( (unsigned __int8)sub_1560180((__int64)v35, 48) )
    {
      v37 = sub_15603C0(v35, -1);
      sub_1E08740(v34, v37);
    }
    else
    {
      sub_1E08740(v34, 0);
    }
  }
  v12 = sub_1E0A0C0(a1);
  v13 = sub_145CBF0((__int64 *)(a1 + 120), 72, 16);
  v14 = *(_QWORD *)(a1 + 16);
  *(_DWORD *)(v13 + 56) = 0;
  *(_QWORD *)(v13 + 64) = v12;
  *(_QWORD *)(a1 + 64) = v13;
  *(_DWORD *)v13 = 1;
  *(_QWORD *)(v13 + 8) = 0;
  *(_QWORD *)(v13 + 16) = 0;
  *(_QWORD *)(v13 + 24) = 0;
  *(_QWORD *)(v13 + 32) = 0;
  *(_QWORD *)(v13 + 40) = 0;
  *(_QWORD *)(v13 + 48) = 0;
  v15 = *(__int64 (**)())(*(_QWORD *)v14 + 56LL);
  if ( v15 == sub_1D12D20 )
    BUG();
  v16 = 34;
  *(_DWORD *)(a1 + 340) = *(_DWORD *)(v15() + 88);
  if ( !(unsigned __int8)sub_1560180(*(_QWORD *)a1 + 112LL, 34) )
  {
    v31 = *(__int64 (**)())(**(_QWORD **)(a1 + 16) + 56LL);
    if ( v31 == sub_1D12D20 )
      BUG();
    v32 = *(_DWORD *)(v31() + 92);
    if ( *(_DWORD *)(a1 + 340) >= v32 )
      v32 = *(_DWORD *)(a1 + 340);
    *(_DWORD *)(a1 + 340) = v32;
  }
  if ( dword_4FC6480 )
    *(_DWORD *)(a1 + 340) = dword_4FC6480;
  v17 = *(_QWORD *)a1;
  *(_QWORD *)(a1 + 72) = 0;
  v18 = 0;
  if ( (*(_BYTE *)(v17 + 18) & 8) != 0 )
    v18 = sub_15E38F0(v17);
  if ( (unsigned int)sub_14DD7D0(v18) - 7 <= 3 )
  {
    v16 = 728;
    v19 = sub_145CBF0((__int64 *)(a1 + 120), 728, 16);
    sub_1F60850(v19);
    *(_QWORD *)(a1 + 88) = v19;
  }
  v20 = 0;
  if ( (*(_BYTE *)(*(_QWORD *)a1 + 18LL) & 8) != 0 )
    v20 = sub_15E38F0(*(_QWORD *)a1);
  v21 = sub_14DD7D0(v20);
  if ( v21 <= 10 )
  {
    if ( v21 <= 6 )
      goto LABEL_24;
  }
  else if ( v21 != 12 )
  {
    goto LABEL_24;
  }
  v16 = 64;
  v33 = (_OWORD *)sub_145CBF0((__int64 *)(a1 + 120), 64, 16);
  *(_QWORD *)(a1 + 80) = v33;
  *v33 = 0;
  v33[1] = 0;
  v33[2] = 0;
  v33[3] = 0;
LABEL_24:
  v22 = 0;
  v23 = *(__int64 (**)(void))(**(_QWORD **)(a1 + 16) + 40LL);
  if ( v23 != sub_1D00B00 )
    v22 = v23();
  result = sub_22077B0(232);
  v29 = result;
  if ( result )
  {
    v16 = v22;
    result = sub_1EB36E0(result, v22);
  }
  v30 = *(_QWORD *)(a1 + 376);
  *(_QWORD *)(a1 + 376) = v29;
  if ( v30 )
    return sub_1E10D30(v30, v16, v25, v26, v27, v28);
  return result;
}
