// Function: sub_1E697B0
// Address: 0x1e697b0
//
__int64 __fastcall sub_1E697B0(_QWORD *a1, int a2)
{
  __int64 v3; // r13
  __int64 *v4; // rdi
  __int64 v5; // rax
  __int64 (*v6)(void); // rdx
  __int64 (*v7)(); // rax
  __int64 v8; // r14
  __int64 v9; // rbx
  __int64 v10; // rdi
  __int64 (__fastcall *v11)(__int64, __int64); // rax
  unsigned __int64 v13; // rbx
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r12
  __int64 v17; // r10
  __int64 v18; // r8
  __int64 v19; // rdi
  unsigned __int64 v20; // rsi
  __int64 (*v21)(); // rax
  __int64 v22; // rax
  __int64 v23; // rax
  unsigned __int64 v24; // [rsp+0h] [rbp-50h]
  __int64 v25; // [rsp+8h] [rbp-48h]
  __int64 v26; // [rsp+10h] [rbp-40h]

  v3 = 0;
  v4 = *(__int64 **)(*a1 + 16LL);
  v5 = *v4;
  v6 = *(__int64 (**)(void))(*v4 + 40);
  if ( v6 != sub_1D00B00 )
  {
    v3 = v6();
    v5 = **(_QWORD **)(*a1 + 16LL);
  }
  v7 = *(__int64 (**)())(v5 + 112);
  if ( v7 == sub_1D00B10 )
    BUG();
  v8 = 16LL * (a2 & 0x7FFFFFFF);
  v9 = *(_QWORD *)(a1[3] + v8);
  v10 = v7();
  v11 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v10 + 160LL);
  if ( v11 == sub_1E693B0 )
    return 0;
  v13 = v9 & 0xFFFFFFFFFFFFFFF8LL;
  v14 = ((__int64 (__fastcall *)(__int64, unsigned __int64, _QWORD))v11)(v10, v13, *a1);
  if ( v13 == v14 )
    return 0;
  if ( a2 < 0 )
  {
    v16 = *(_QWORD *)(a1[3] + v8 + 8);
  }
  else
  {
    v15 = a1[34];
    v16 = *(_QWORD *)(v15 + 8LL * (unsigned int)a2);
  }
  if ( v16 )
  {
    if ( (*(_BYTE *)(v16 + 4) & 8) != 0 )
    {
      while ( 1 )
      {
        v16 = *(_QWORD *)(v16 + 32);
        if ( !v16 )
          break;
        if ( (*(_BYTE *)(v16 + 4) & 8) == 0 )
          goto LABEL_11;
      }
    }
    else
    {
LABEL_11:
      v17 = *(_QWORD *)(v16 + 16);
      v18 = 0;
      v19 = *(_QWORD *)(*a1 + 16LL);
      v20 = 0xCCCCCCCCCCCCCCCDLL * ((v16 - *(_QWORD *)(v17 + 32)) >> 3);
      v21 = *(__int64 (**)())(*(_QWORD *)v19 + 112LL);
      if ( v21 != sub_1D00B10 )
      {
        v24 = 0xCCCCCCCCCCCCCCCDLL * ((v16 - *(_QWORD *)(v17 + 32)) >> 3);
        v25 = *(_QWORD *)(v16 + 16);
        v26 = v14;
        v23 = ((__int64 (__fastcall *)(__int64, unsigned __int64, __int64, __int64, _QWORD))v21)(v19, v20, v14, v15, 0);
        v20 = v24;
        v17 = v25;
        v14 = v26;
        v18 = v23;
      }
      v22 = sub_1E16EE0(v17, v20, v14, v3, v18);
      v14 = v22;
      if ( !v22 || v22 == v13 )
        return 0;
      while ( 1 )
      {
        v16 = *(_QWORD *)(v16 + 32);
        if ( !v16 )
          break;
        if ( (*(_BYTE *)(v16 + 4) & 8) == 0 )
          goto LABEL_11;
      }
    }
  }
  sub_1E693D0((__int64)a1, a2, v14);
  return 1;
}
