// Function: sub_1949170
// Address: 0x1949170
//
__int64 __fastcall sub_1949170(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _BYTE *a5)
{
  __int64 v8; // r8
  bool v9; // zf
  _QWORD *v10; // rax
  __int64 v11; // r12
  __int64 v12; // r14
  _QWORD *v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r13
  __int64 v17; // rdi
  int v18; // r10d
  int v19; // r11d
  __int64 v20; // rax
  __int64 v22; // [rsp+8h] [rbp-58h]
  _BYTE *v23; // [rsp+10h] [rbp-50h] BYREF
  __int16 v24; // [rsp+20h] [rbp-40h]

  v8 = *(_QWORD *)(a3 + 8);
  v9 = *a5 == 0;
  v24 = 257;
  if ( !v9 )
  {
    v23 = a5;
    LOBYTE(v24) = 3;
  }
  v22 = v8;
  v10 = (_QWORD *)sub_22077B0(64);
  v11 = (__int64)v10;
  if ( v10 )
    sub_157FB60(v10, a2, (__int64)&v23, a1, v22);
  v12 = *(_QWORD *)(a3 + 8);
  v13 = sub_1648A60(56, 1u);
  if ( v13 )
    sub_15F8590((__int64)v13, v12, v11);
  v14 = sub_157F280(*(_QWORD *)(a3 + 8));
  v16 = v15;
  v17 = v14;
  while ( v16 != v17 )
  {
    if ( (*(_DWORD *)(v17 + 20) & 0xFFFFFFF) != 0 )
    {
      do
        sub_1948E70(v17, a4, v11);
      while ( v19 != v18 );
    }
    v20 = *(_QWORD *)(v17 + 32);
    if ( !v20 )
      BUG();
    v17 = 0;
    if ( *(_BYTE *)(v20 - 8) == 77 )
      v17 = v20 - 24;
  }
  return v11;
}
