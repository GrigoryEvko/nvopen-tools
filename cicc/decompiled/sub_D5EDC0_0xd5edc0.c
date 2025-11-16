// Function: sub_D5EDC0
// Address: 0xd5edc0
//
__int64 __fastcall sub_D5EDC0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  int v4; // edx
  __int64 v5; // r13
  unsigned int v6; // r14d
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // r14
  char v12; // bl
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rbx
  __int64 v18; // r13
  __int64 v20; // rbx
  __int64 v21; // r14
  __int64 v22; // rdx
  unsigned int v23; // esi
  __int64 v24; // [rsp+8h] [rbp-A8h]
  _BYTE v25[32]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v26; // [rsp+40h] [rbp-70h]
  _QWORD v27[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v28; // [rsp+70h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 72);
  v4 = *(unsigned __int8 *)(v3 + 8);
  if ( (_BYTE)v4 != 12
    && (unsigned __int8)v4 > 3u
    && (_BYTE)v4 != 5
    && (v4 & 0xFB) != 0xA
    && (v4 & 0xFD) != 4
    && ((unsigned __int8)(v4 - 15) > 3u && v4 != 20 || !(unsigned __int8)sub_BCEBA0(v3, 0)) )
  {
    return 0;
  }
  v5 = *(_QWORD *)a1;
  v28 = 257;
  v6 = *(_DWORD *)(v5 + 4);
  v7 = sub_BD5C60(a2);
  v8 = sub_AE4540(v5, v7, v6);
  v9 = sub_A830B0((unsigned int **)(a1 + 24), *(_QWORD *)(a2 - 32), v8, (__int64)v27);
  v10 = *(_QWORD *)a1;
  v11 = v9;
  v24 = *(_QWORD *)(a2 + 72);
  v12 = sub_AE5020(*(_QWORD *)a1, v24);
  v13 = sub_9208B0(v10, v24);
  v27[1] = v14;
  v27[0] = v13;
  v15 = sub_B33F60(
          a1 + 24,
          *(_QWORD *)(v11 + 8),
          ((1LL << v12) + ((unsigned __int64)(v13 + 7) >> 3) - 1) >> v12 << v12,
          v14);
  v16 = *(_QWORD *)(a1 + 104);
  v17 = v15;
  v26 = 257;
  v18 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v16 + 32LL))(
          v16,
          17,
          v15,
          v11,
          0,
          0);
  if ( !v18 )
  {
    v28 = 257;
    v18 = sub_B504D0(17, v17, v11, (__int64)v27, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 112) + 16LL))(
      *(_QWORD *)(a1 + 112),
      v18,
      v25,
      *(_QWORD *)(a1 + 80),
      *(_QWORD *)(a1 + 88));
    v20 = *(_QWORD *)(a1 + 24);
    v21 = v20 + 16LL * *(unsigned int *)(a1 + 32);
    while ( v21 != v20 )
    {
      v22 = *(_QWORD *)(v20 + 8);
      v23 = *(_DWORD *)v20;
      v20 += 16;
      sub_B99FD0(v18, v23, v22);
    }
  }
  return v18;
}
