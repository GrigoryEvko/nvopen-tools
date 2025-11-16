// Function: sub_1E42180
// Address: 0x1e42180
//
bool __fastcall sub_1E42180(_QWORD *a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v5; // rax
  unsigned __int64 v7; // rdi
  __int64 v8; // r12
  __int64 v9; // r13
  __int64 v10; // rcx
  __int64 v11; // rax
  __int16 v12; // dx
  bool v13; // al
  __int64 v14; // rax
  __int16 v15; // dx
  char v16; // al
  __int64 v17; // rcx
  __int64 v18; // r14
  __int64 (*v19)(void); // rax
  __int64 v20; // rdi
  __int64 (*v21)(); // rax
  __int64 v22; // rax
  __int64 v23; // rsi
  unsigned int i; // edx
  __int64 v25; // rdi
  __int64 v26; // rax
  unsigned int v27; // [rsp+Ch] [rbp-44h] BYREF
  unsigned int v28; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v29; // [rsp+14h] [rbp-3Ch] BYREF
  int v30; // [rsp+18h] [rbp-38h] BYREF
  int v31; // [rsp+1Ch] [rbp-34h] BYREF
  __int64 v32; // [rsp+20h] [rbp-30h] BYREF
  _QWORD v33[5]; // [rsp+28h] [rbp-28h] BYREF

  v5 = (*(__int64 *)a3 >> 1) & 3;
  if ( v5 == 3 )
  {
    if ( *(_DWORD *)(a3 + 8) == 3 )
      return 0;
  }
  else if ( v5 != 2 )
  {
    return 0;
  }
  if ( byte_4FC6D60 != 1 || v5 == 2 )
    return 1;
  v7 = *(_QWORD *)a3 & 0xFFFFFFFFFFFFFFF8LL;
  v8 = *(_QWORD *)(a2 + 8);
  v9 = *(_QWORD *)(v7 + 8);
  if ( !a4 )
  {
    v8 = *(_QWORD *)(v7 + 8);
    v9 = *(_QWORD *)(a2 + 8);
  }
  if ( sub_1E17880(v8) || sub_1E17880(v9) || (unsigned __int8)sub_1E178F0(v8) || (unsigned __int8)sub_1E178F0(v9) )
    return 1;
  v11 = *(_QWORD *)(v9 + 16);
  if ( *(_WORD *)v11 != 1 || (*(_BYTE *)(*(_QWORD *)(v9 + 32) + 64LL) & 0x10) == 0 )
  {
    v12 = *(_WORD *)(v9 + 46);
    if ( (v12 & 4) != 0 || (v12 & 8) == 0 )
      v13 = (*(_QWORD *)(v11 + 8) & 0x20000LL) != 0;
    else
      v13 = sub_1E15D00(v9, 0x20000u, 1);
    if ( !v13 )
      return 0;
  }
  v14 = *(_QWORD *)(v8 + 16);
  if ( *(_WORD *)v14 != 1 || (*(_BYTE *)(*(_QWORD *)(v8 + 32) + 64LL) & 8) == 0 )
  {
    v15 = *(_WORD *)(v8 + 46);
    if ( (v15 & 4) != 0 || (v15 & 8) == 0 )
      v16 = WORD1(*(_QWORD *)(v14 + 8)) & 1;
    else
      v16 = sub_1E15D00(v8, 0x10000u, 1);
    if ( !v16 )
      return 0;
  }
  if ( !(unsigned __int8)sub_1E41020((__int64)a1, v8, (int *)&v27, v10)
    || !(unsigned __int8)sub_1E41020((__int64)a1, v9, (int *)&v28, v17) )
  {
    return 1;
  }
  v18 = 0;
  v19 = *(__int64 (**)(void))(**(_QWORD **)(a1[4] + 16LL) + 112LL);
  if ( v19 != sub_1D00B10 )
    v18 = v19();
  v20 = a1[2];
  v21 = *(__int64 (**)())(*(_QWORD *)v20 + 592LL);
  if ( v21 == sub_1D9BA90 )
    return 1;
  if ( !((unsigned __int8 (__fastcall *)(__int64, __int64, unsigned int *, __int64 *, __int64))v21)(
          v20,
          v8,
          &v29,
          &v32,
          v18) )
    return 1;
  if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64, int *, _QWORD *, __int64))(*(_QWORD *)a1[2] + 592LL))(
          a1[2],
          v9,
          &v30,
          v33,
          v18) )
    return 1;
  if ( v29 != v30 )
    return 1;
  v22 = sub_1E69D00(a1[5], v29);
  if ( !v22 || **(_WORD **)(v22 + 16) != 45 && **(_WORD **)(v22 + 16) )
    return 1;
  v23 = 0;
  for ( i = 1; *(_DWORD *)(v22 + 40) != i; i += 2 )
  {
    v25 = *(_QWORD *)(v22 + 32);
    if ( a1[115] == *(_QWORD *)(v25 + 40LL * (i + 1) + 24) )
      v23 = *(unsigned int *)(v25 + 40LL * i + 8);
  }
  v26 = sub_1E69D00(a1[5], v23);
  v31 = 0;
  if ( !v26 || !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64, int *))(*(_QWORD *)a1[2] + 608LL))(a1[2], v26, &v31) )
    return 1;
  if ( v32 < v33[0] )
    return *(_QWORD *)(**(_QWORD **)(v9 + 56) + 24LL) + v33[0] > (unsigned __int64)v28;
  return *(_QWORD *)(**(_QWORD **)(v8 + 56) + 24LL) + v32 > (unsigned __int64)v27;
}
