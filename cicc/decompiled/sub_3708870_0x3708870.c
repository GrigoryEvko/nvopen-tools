// Function: sub_3708870
// Address: 0x3708870
//
unsigned __int64 *__fastcall sub_3708870(unsigned __int64 *a1, __int64 a2, _WORD *a3)
{
  int v4; // r8d
  __int16 v5; // ax
  __int16 v6; // dx
  unsigned __int64 v8; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v9[6]; // [rsp+10h] [rbp-30h] BYREF

  v9[0] = 0;
  v9[1] = 0;
  sub_1254950(&v8, a2, (__int64)v9, 2u);
  if ( (v8 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v8 & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
  else
  {
    v4 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 24) + 16LL))(*(_QWORD *)(a2 + 24));
    v5 = *(_WORD *)v9[0];
    v6 = __ROL2__(*(_WORD *)v9[0], 8);
    if ( v4 != 1 )
      v5 = v6;
    *a3 = v5;
    *a1 = 1;
    return a1;
  }
}
