// Function: sub_3910390
// Address: 0x3910390
//
__int64 __fastcall sub_3910390(__int64 a1, _QWORD *a2)
{
  __int64 v3; // r13
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 *v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rax
  _QWORD v13[2]; // [rsp+0h] [rbp-50h] BYREF
  char v14; // [rsp+10h] [rbp-40h]
  char v15; // [rsp+11h] [rbp-3Fh]

  v3 = a2[1];
  v15 = 1;
  v13[0] = "strtab_begin";
  v14 = 3;
  v4 = sub_38BF8E0(v3, (__int64)v13, 0, 1);
  v15 = 1;
  v13[0] = "strtab_end";
  v14 = 3;
  v5 = sub_38BF8E0(v3, (__int64)v13, 0, 1);
  (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a2 + 424LL))(a2, 243, 4);
  (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64))(*a2 + 688LL))(a2, v5, v4, 4);
  (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a2 + 176LL))(a2, v4, 0);
  if ( !*(_BYTE *)(a1 + 64) )
  {
    v7 = sub_390FEA0(a1);
    sub_38D4150((__int64)a2, v7, 0);
    v8 = *((unsigned int *)a2 + 30);
    v9 = 0;
    if ( (_DWORD)v8 )
      v9 = *(_QWORD *)(a2[14] + 32 * v8 - 32);
    v10 = (__int64 *)a2[34];
    v11 = *v10;
    v12 = *(_QWORD *)v7 & 7LL;
    *(_QWORD *)(v7 + 8) = v10;
    v11 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v7 = v11 | v12;
    *(_QWORD *)(v11 + 8) = v7;
    *v10 = v7 | *v10 & 7;
    *(_QWORD *)(v7 + 24) = v9;
    *(_BYTE *)(a1 + 64) = 1;
  }
  (*(void (__fastcall **)(_QWORD *, __int64, _QWORD, __int64, _QWORD))(*a2 + 512LL))(a2, 4, 0, 1, 0);
  return (*(__int64 (__fastcall **)(_QWORD *, __int64, _QWORD))(*a2 + 176LL))(a2, v5, 0);
}
