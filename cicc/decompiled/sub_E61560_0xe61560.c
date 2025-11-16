// Function: sub_E61560
// Address: 0xe61560
//
__int64 __fastcall sub_E61560(__int64 a1, _QWORD *a2)
{
  _QWORD *v3; // r15
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v7; // rax
  unsigned __int64 v8; // r14
  __int64 v9; // rdx
  _QWORD v10[4]; // [rsp+0h] [rbp-60h] BYREF
  char v11; // [rsp+20h] [rbp-40h]
  char v12; // [rsp+21h] [rbp-3Fh]

  v3 = (_QWORD *)a2[1];
  v12 = 1;
  v10[0] = "strtab_begin";
  v11 = 3;
  v4 = sub_E6C380(v3, v10, 0);
  v12 = 1;
  v10[0] = "strtab_end";
  v11 = 3;
  v5 = sub_E6C380(v3, v10, 0);
  (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a2 + 536LL))(a2, 243, 4);
  (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64))(*a2 + 832LL))(a2, v5, v4, 4);
  (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a2 + 208LL))(a2, v4, 0);
  if ( !*(_QWORD *)(a1 + 32) )
  {
    v7 = v3[36];
    v3[46] += 208LL;
    v8 = (v7 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v3[37] >= v8 + 208 && v7 )
      v3[36] = v8 + 208;
    else
      v8 = sub_9D1E70((__int64)(v3 + 36), 208, 208, 3);
    sub_E81B30(v8, 1, 0);
    *(_BYTE *)(v8 + 30) = 0;
    *(_QWORD *)(v8 + 40) = v8 + 64;
    *(_QWORD *)(v8 + 96) = v8 + 112;
    *(_QWORD *)(v8 + 32) = 0;
    *(_QWORD *)(v8 + 48) = 0;
    *(_QWORD *)(v8 + 56) = 32;
    *(_QWORD *)(v8 + 104) = 0x400000000LL;
    *(_QWORD *)(a1 + 32) = v8;
    v9 = *(_QWORD *)(a2[36] + 8LL);
    *(_QWORD *)(v8 + 8) = v9;
    *(_DWORD *)(v8 + 24) = *(_DWORD *)(a2[36] + 24LL) + 1;
    *(_QWORD *)a2[36] = v8;
    a2[36] = v8;
    *(_QWORD *)(*(_QWORD *)(v9 + 8) + 8LL) = v8;
  }
  (*(void (__fastcall **)(_QWORD *, __int64, _QWORD, __int64, _QWORD))(*a2 + 608LL))(a2, 2, 0, 1, 0);
  return (*(__int64 (__fastcall **)(_QWORD *, __int64, _QWORD))(*a2 + 208LL))(a2, v5, 0);
}
