// Function: sub_D64C90
// Address: 0xd64c90
//
__int64 __fastcall sub_D64C90(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r12
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rbx
  __int64 v11; // r14
  __int64 v12; // rbx
  __int64 v13; // r13
  __int64 v14; // rdx
  unsigned int v15; // esi
  char v16[32]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v17; // [rsp+30h] [rbp-70h]
  char v18[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v19; // [rsp+60h] [rbp-40h]

  v3 = sub_D63080(a1, *(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( !v4 )
    return 0;
  v5 = v3;
  if ( !v3 )
    return 0;
  v6 = v4;
  v7 = sub_1028510(a1 + 24, *(_QWORD *)a1, a2, 1);
  v8 = *(_QWORD *)(a1 + 104);
  v9 = v7;
  v17 = 257;
  if ( !(*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v8 + 32LL))(
          v8,
          13,
          v6,
          v7,
          0,
          0) )
  {
    v19 = 257;
    v11 = sub_B504D0(13, v6, v9, (__int64)v18, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, char *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 112) + 16LL))(
      *(_QWORD *)(a1 + 112),
      v11,
      v16,
      *(_QWORD *)(a1 + 80),
      *(_QWORD *)(a1 + 88));
    v12 = *(_QWORD *)(a1 + 24);
    v13 = v12 + 16LL * *(unsigned int *)(a1 + 32);
    while ( v13 != v12 )
    {
      v14 = *(_QWORD *)(v12 + 8);
      v15 = *(_DWORD *)v12;
      v12 += 16;
      sub_B99FD0(v11, v15, v14);
    }
  }
  return v5;
}
