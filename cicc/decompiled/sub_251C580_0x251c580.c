// Function: sub_251C580
// Address: 0x251c580
//
__int64 __fastcall sub_251C580(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // r8
  __int64 v5; // r9
  unsigned __int64 v6; // rcx
  __int64 v7; // r13
  __int64 v8; // rax
  unsigned int v9; // r14d
  __int64 v10; // r15
  __int64 (*v11)(); // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v17; // rax
  __int64 (*v18)(); // rax
  char v19; // [rsp+1Fh] [rbp-101h] BYREF
  _BYTE *v20; // [rsp+20h] [rbp-100h] BYREF
  __int64 v21; // [rsp+28h] [rbp-F8h]
  _BYTE v22[240]; // [rsp+30h] [rbp-F0h] BYREF

  v2 = a2;
  v20 = (_BYTE *)a2;
  v3 = sub_C99770("updateAA", 8, (void (__fastcall *)(__m128i **, __int64))sub_2509250, (__int64)&v20);
  v6 = *(unsigned int *)(a1 + 412);
  v7 = v3;
  v20 = v22;
  v21 = 0x800000000LL;
  v8 = *(unsigned int *)(a1 + 408);
  if ( v8 + 1 > v6 )
  {
    sub_C8D5F0(a1 + 400, (const void *)(a1 + 416), v8 + 1, 8u, v4, v5);
    v8 = *(unsigned int *)(a1 + 408);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 400) + 8 * v8) = &v20;
  v9 = 1;
  ++*(_DWORD *)(a1 + 408);
  v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 40LL))(a2);
  v19 = 0;
  if ( !(unsigned __int8)sub_251C440(a1, a2, 0, &v19, 1, 1) )
  {
    a2 = a1;
    v9 = sub_250D1D0(v2, a1);
  }
  v11 = *(__int64 (**)())(*(_QWORD *)v2 + 32LL);
  if ( (v11 == sub_2505D80 || !((unsigned __int8 (__fastcall *)(__int64))v11)(v2)) && !(_DWORD)v21 )
  {
    v17 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 40LL))(v2);
    if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v17 + 24LL))(v17) )
    {
      if ( v9 || (a2 = a1, (unsigned int)sub_250D1D0(v2, a1) == 1) )
      {
        v18 = *(__int64 (**)())(*(_QWORD *)v2 + 32LL);
        if ( (v18 == sub_2505D80 || !((unsigned __int8 (__fastcall *)(__int64))v18)(v2)) && !(_DWORD)v21 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v10 + 32LL))(v10);
      }
    }
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v10 + 24LL))(v10) )
    sub_251BB40(a1, a2, v12, v13, v14, v15);
  --*(_DWORD *)(a1 + 408);
  if ( v20 != v22 )
    _libc_free((unsigned __int64)v20);
  if ( v7 )
    sub_C9AF60(v7);
  return v9;
}
