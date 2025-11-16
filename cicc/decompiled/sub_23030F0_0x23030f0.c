// Function: sub_23030F0
// Address: 0x23030f0
//
__int64 __fastcall sub_23030F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v5; // r14
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // r13
  __int64 v8; // r13
  _QWORD *v9; // rdi

  v4 = *(_QWORD *)(a3 + 80);
  if ( !v4 )
  {
    sub_B2BE50(a3);
    BUG();
  }
  v5 = sub_B2BE50(a3);
  v6 = *(_QWORD *)(v4 + 24) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v6 == v4 + 24 )
  {
    v7 = 0;
  }
  else
  {
    if ( !v6 )
      BUG();
    v7 = v6 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v6 - 24) - 30 >= 0xB )
      v7 = 0;
  }
  v8 = v7 + 24;
  v9 = sub_BD2C40(72, unk_3F148B8);
  if ( v9 )
    sub_B4C8A0((__int64)v9, v5, v8, 0);
  memset((void *)a1, 0, 0x60u);
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_DWORD *)(a1 + 16) = 2;
  *(_DWORD *)(a1 + 64) = 2;
  *(_BYTE *)(a1 + 76) = 1;
  return a1;
}
