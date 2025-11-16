// Function: sub_30476E0
// Address: 0x30476e0
//
__int64 __fastcall sub_30476E0(__int64 a1, __int64 a2, unsigned int a3, int a4, __int64 a5, int a6)
{
  __int64 v9; // rsi
  unsigned __int16 *v10; // rdx
  int v11; // esi
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // r13
  __int64 v16; // [rsp+0h] [rbp-30h] BYREF
  int v17; // [rsp+8h] [rbp-28h]

  v9 = *(_QWORD *)(a2 + 80);
  v16 = v9;
  if ( v9 )
    sub_B96E90((__int64)&v16, v9, 1);
  v10 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * a3);
  v17 = *(_DWORD *)(a2 + 72);
  if ( *v10 == 7 )
  {
    v11 = 5590;
  }
  else
  {
    if ( *v10 != 8 )
      BUG();
    v11 = 5591;
  }
  v12 = sub_33F77A0(
          a4,
          v11,
          (unsigned int)&v16,
          *v10,
          *((_QWORD *)v10 + 1),
          a6,
          *(_OWORD *)*(_QWORD *)(a2 + 40),
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
  v13 = v16;
  v14 = v12;
  *(_DWORD *)(v12 + 72) = *(_DWORD *)(a2 + 72);
  if ( v13 )
    sub_B91220((__int64)&v16, v13);
  return v14;
}
