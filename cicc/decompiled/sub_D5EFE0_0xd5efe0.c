// Function: sub_D5EFE0
// Address: 0xd5efe0
//
__int64 __fastcall sub_D5EFE0(__int64 a1, __int64 a2)
{
  int v3; // edx
  __int64 v4; // r13
  __int64 result; // rax
  int v6; // eax
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // r13
  __int64 v10; // r12
  __int64 v11; // r13
  __int64 v12; // r14
  __int64 v13; // rdx
  unsigned int v14; // esi
  __int64 v15; // [rsp+8h] [rbp-B8h]
  __m128i v16; // [rsp+10h] [rbp-B0h] BYREF
  char v17; // [rsp+28h] [rbp-98h]
  _BYTE v18[32]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v19; // [rsp+50h] [rbp-70h]
  _BYTE v20[32]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v21; // [rsp+80h] [rbp-40h]

  sub_D5BE70(&v16, (unsigned __int8 *)a2, *(__int64 **)(a1 + 8));
  if ( !v17 || v16.m128i_i8[0] == 4 )
    return 0;
  v3 = *(_DWORD *)(a2 + 4);
  v4 = v16.m128i_i32[3];
  v21 = 257;
  result = sub_A830B0(
             (unsigned int **)(a1 + 24),
             *(_QWORD *)(a2 + 32 * (v16.m128i_u32[2] - (unsigned __int64)(v3 & 0x7FFFFFF))),
             *(_QWORD *)(a1 + 208),
             (__int64)v20);
  if ( (int)v4 >= 0 )
  {
    v15 = result;
    v6 = *(_DWORD *)(a2 + 4);
    v21 = 257;
    v7 = sub_A830B0(
           (unsigned int **)(a1 + 24),
           *(_QWORD *)(a2 + 32 * (v4 - (v6 & 0x7FFFFFF))),
           *(_QWORD *)(a1 + 208),
           (__int64)v20);
    v8 = *(_QWORD *)(a1 + 104);
    v9 = v7;
    v19 = 257;
    v10 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v8 + 32LL))(
            v8,
            17,
            v15,
            v7,
            0,
            0);
    if ( !v10 )
    {
      v21 = 257;
      v10 = sub_B504D0(17, v15, v9, (__int64)v20, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 112) + 16LL))(
        *(_QWORD *)(a1 + 112),
        v10,
        v18,
        *(_QWORD *)(a1 + 80),
        *(_QWORD *)(a1 + 88));
      v11 = *(_QWORD *)(a1 + 24);
      v12 = v11 + 16LL * *(unsigned int *)(a1 + 32);
      while ( v12 != v11 )
      {
        v13 = *(_QWORD *)(v11 + 8);
        v14 = *(_DWORD *)v11;
        v11 += 16;
        sub_B99FD0(v10, v14, v13);
      }
    }
    return v10;
  }
  return result;
}
