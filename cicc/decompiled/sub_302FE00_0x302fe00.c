// Function: sub_302FE00
// Address: 0x302fe00
//
__int64 __fastcall sub_302FE00(__int64 a1, __int64 a2, _BYTE *a3, int a4)
{
  __int64 v4; // r12
  _QWORD *v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // rdx
  unsigned int v9; // esi
  _QWORD *v11; // rax
  __int64 v12; // rbx
  __int64 v13; // r13
  __int64 v14; // rdx
  unsigned int v15; // esi
  _BYTE v16[32]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v17; // [rsp+20h] [rbp-30h]

  if ( *a3 != 65 )
    return sub_2FE3F70(a1, (__int64 *)a2, a3, a4);
  v4 = 0;
  if ( !byte_3F8E4E0[8 * a4 + 5] )
    return v4;
  v17 = 257;
  if ( a4 != 7 )
  {
    v5 = sub_BD2C40(80, unk_3F222C8);
    v4 = (__int64)v5;
    if ( v5 )
      sub_B4D930((__int64)v5, *(_QWORD *)(a2 + 72), 5, 1, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
      *(_QWORD *)(a2 + 88),
      v4,
      v16,
      *(_QWORD *)(a2 + 56),
      *(_QWORD *)(a2 + 64));
    v6 = *(_QWORD *)a2;
    v7 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
    if ( *(_QWORD *)a2 != v7 )
    {
      do
      {
        v8 = *(_QWORD *)(v6 + 8);
        v9 = *(_DWORD *)v6;
        v6 += 16;
        sub_B99FD0(v4, v9, v8);
      }
      while ( v7 != v6 );
    }
    return v4;
  }
  v11 = sub_BD2C40(80, unk_3F222C8);
  v4 = (__int64)v11;
  if ( v11 )
    sub_B4D930((__int64)v11, *(_QWORD *)(a2 + 72), 7, 1, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
    *(_QWORD *)(a2 + 88),
    v4,
    v16,
    *(_QWORD *)(a2 + 56),
    *(_QWORD *)(a2 + 64));
  v12 = *(_QWORD *)a2;
  v13 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 == v13 )
    return v4;
  do
  {
    v14 = *(_QWORD *)(v12 + 8);
    v15 = *(_DWORD *)v12;
    v12 += 16;
    sub_B99FD0(v4, v15, v14);
  }
  while ( v13 != v12 );
  return v4;
}
