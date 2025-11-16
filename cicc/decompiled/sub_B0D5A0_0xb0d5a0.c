// Function: sub_B0D5A0
// Address: 0xb0d5a0
//
__int64 __fastcall sub_B0D5A0(__int64 a1, _DWORD *a2)
{
  unsigned __int64 v4; // rax
  __int64 *v5; // rsi
  __int64 *v7; // rdi
  __int64 *v8; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int64 v9; // [rsp+8h] [rbp-28h]
  char v10; // [rsp+10h] [rbp-20h]

  sub_AF4640((__int64)&v8, a1);
  if ( !v10 )
    return 0;
  v4 = v9;
  if ( v9 <= 3 )
    return a1;
  v5 = v8;
  if ( *v8 != 16 || v8[2] != 22 || v8[3] != 24 )
    return a1;
  *a2 = v8[1];
  if ( v4 == 4 )
    return 0;
  v7 = (__int64 *)(*(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a1 + 8) & 4) != 0 )
    v7 = (__int64 *)*v7;
  return sub_B0D000(v7, v5, v4 - 4, 0, 1);
}
