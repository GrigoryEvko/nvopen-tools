// Function: sub_3749B60
// Address: 0x3749b60
//
__int64 __fastcall sub_3749B60(__int64 *a1, __int64 a2)
{
  __int64 *v4; // rdx
  __int32 v5; // ebx
  __int64 v7; // rdx
  unsigned int v8; // eax
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 *v11; // rdi
  __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 (__fastcall *v14)(__int64, unsigned __int16); // rcx
  unsigned __int32 v15; // r14d
  _QWORD *v16; // rax
  __int64 v17; // rdx
  __m128i v18; // [rsp+0h] [rbp-50h] BYREF
  __int64 v19; // [rsp+10h] [rbp-40h]
  __int64 v20; // [rsp+18h] [rbp-38h]
  __int64 v21; // [rsp+20h] [rbp-30h]

  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v4 = *(__int64 **)(a2 - 8);
  else
    v4 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v5 = sub_3746830(a1, *v4);
  if ( !v5 )
    return 0;
  v7 = (*(_BYTE *)(a2 + 7) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v8 = sub_2D5BAE0(a1[16], a1[14], *(__int64 **)(*(_QWORD *)v7 + 8LL), 0);
  if ( (unsigned __int16)v8 <= 1u )
    return 0;
  v11 = (__int64 *)a1[16];
  v12 = v11[(unsigned __int16)v8 + 14];
  if ( !v12 )
    return 0;
  v13 = *v11;
  v14 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*v11 + 552);
  if ( v14 != sub_2EC09E0 )
    v12 = ((__int64 (__fastcall *)(__int64 *, _QWORD, _QWORD))v14)(v11, v8, 0);
  v15 = sub_3741980((__int64)a1, v12, v13, (__int64)v14, v9, v10);
  v16 = sub_2F26260(*(_QWORD *)(a1[5] + 744), *(__int64 **)(a1[5] + 752), a1 + 10, *(_QWORD *)(a1[15] + 8) - 800LL, v15);
  v18.m128i_i64[0] = 0;
  v19 = 0;
  v18.m128i_i32[2] = v5;
  v20 = 0;
  v21 = 0;
  sub_2E8EAD0(v17, (__int64)v16, &v18);
  sub_3742B00((__int64)a1, (_BYTE *)a2, v15, 1);
  return 1;
}
