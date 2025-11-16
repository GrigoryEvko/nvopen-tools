// Function: sub_F8D260
// Address: 0xf8d260
//
__int64 __fastcall sub_F8D260(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  char v3; // al
  __int64 v4; // rbx
  char v5; // r14
  __int64 v6; // r12
  _QWORD *v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r13
  __int64 v11; // rdx
  unsigned int v12; // esi
  _BYTE v13[32]; // [rsp+0h] [rbp-90h] BYREF
  __int16 v14; // [rsp+20h] [rbp-70h]
  _BYTE v15[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v16; // [rsp+50h] [rbp-40h]

  v2 = sub_F894B0(a1, *(_QWORD *)(a2 + 32));
  v3 = sub_DBED40(*(_QWORD *)a1, *(_QWORD *)(a2 + 32));
  v4 = *(_QWORD *)(a2 + 40);
  v14 = 257;
  if ( v4 == *(_QWORD *)(v2 + 8) )
    return v2;
  v5 = v3;
  v6 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a1 + 600) + 120LL))(
         *(_QWORD *)(a1 + 600),
         39,
         v2,
         v4);
  if ( !v6 )
  {
    v16 = 257;
    v8 = sub_BD2C40(72, unk_3F10A14);
    v6 = (__int64)v8;
    if ( v8 )
      sub_B515B0((__int64)v8, v2, v4, (__int64)v15, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 608) + 16LL))(
      *(_QWORD *)(a1 + 608),
      v6,
      v13,
      *(_QWORD *)(a1 + 576),
      *(_QWORD *)(a1 + 584));
    v9 = *(_QWORD *)(a1 + 520);
    v10 = v9 + 16LL * *(unsigned int *)(a1 + 528);
    while ( v10 != v9 )
    {
      v11 = *(_QWORD *)(v9 + 8);
      v12 = *(_DWORD *)v9;
      v9 += 16;
      sub_B99FD0(v6, v12, v11);
    }
    if ( v5 )
      sub_B448D0(v6, 1);
  }
  return v6;
}
