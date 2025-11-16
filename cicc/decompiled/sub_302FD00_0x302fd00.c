// Function: sub_302FD00
// Address: 0x302fd00
//
__int64 __fastcall sub_302FD00(__int64 a1, __int64 a2, _BYTE *a3, int a4)
{
  __int64 v4; // r12
  _QWORD *v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // rdx
  unsigned int v9; // esi
  _BYTE v11[32]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v12; // [rsp+20h] [rbp-30h]

  if ( *a3 != 65 )
    return sub_2FE4100(a1, (__int64 *)a2, (__int64)a3, a4);
  v4 = 0;
  if ( byte_3F8E4E0[8 * a4 + 4] && (*(_DWORD *)(*(_QWORD *)(*((_QWORD *)a3 - 8) + 8LL) + 8LL) <= 0x1FFFu || a4 != 7) )
  {
    v12 = 257;
    v5 = sub_BD2C40(80, unk_3F222C8);
    v4 = (__int64)v5;
    if ( v5 )
      sub_B4D930((__int64)v5, *(_QWORD *)(a2 + 72), 4, 1, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
      *(_QWORD *)(a2 + 88),
      v4,
      v11,
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
  }
  return v4;
}
