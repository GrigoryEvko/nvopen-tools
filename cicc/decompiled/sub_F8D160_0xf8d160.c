// Function: sub_F8D160
// Address: 0xf8d160
//
__int64 __fastcall sub_F8D160(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r15
  __int64 v4; // r14
  __int64 v5; // r12
  __int64 v7; // rbx
  __int64 v8; // r13
  __int64 v9; // rdx
  unsigned int v10; // esi
  _BYTE v11[32]; // [rsp+0h] [rbp-90h] BYREF
  __int16 v12; // [rsp+20h] [rbp-70h]
  _BYTE v13[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v14; // [rsp+50h] [rbp-40h]

  v2 = sub_F894B0(a1, *(_QWORD *)(a2 + 32));
  v3 = *(_QWORD *)(a2 + 40);
  v12 = 257;
  v4 = v2;
  if ( v3 == *(_QWORD *)(v2 + 8) )
    return v2;
  v5 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a1 + 600) + 120LL))(
         *(_QWORD *)(a1 + 600),
         38,
         v2,
         v3);
  if ( !v5 )
  {
    v14 = 257;
    v5 = sub_B51D30(38, v4, v3, (__int64)v13, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 608) + 16LL))(
      *(_QWORD *)(a1 + 608),
      v5,
      v11,
      *(_QWORD *)(a1 + 576),
      *(_QWORD *)(a1 + 584));
    v7 = *(_QWORD *)(a1 + 520);
    v8 = v7 + 16LL * *(unsigned int *)(a1 + 528);
    while ( v8 != v7 )
    {
      v9 = *(_QWORD *)(v7 + 8);
      v10 = *(_DWORD *)v7;
      v7 += 16;
      sub_B99FD0(v5, v10, v9);
    }
  }
  return v5;
}
