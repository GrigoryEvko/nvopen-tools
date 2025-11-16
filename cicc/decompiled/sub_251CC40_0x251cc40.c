// Function: sub_251CC40
// Address: 0x251cc40
//
__int64 __fastcall sub_251CC40(
        __int64 a1,
        __int64 (__fastcall *a2)(__int64, unsigned __int64 *, __int64),
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5)
{
  unsigned __int64 v9; // rsi
  __int64 v10; // r13
  __int64 v11; // rax
  unsigned __int64 v12[6]; // [rsp+0h] [rbp-30h] BYREF

  if ( **(_BYTE **)(a5 - 32) )
  {
    sub_250D230(v12, a5, 5, 0);
    v9 = v12[0];
    v10 = sub_251C7D0(a1, v12[0], v12[1], a4, 1, 0, 1);
    if ( !v10 || (*(unsigned __int8 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v10 + 120LL))(v10, v9) )
    {
      return 0;
    }
    else
    {
      v11 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 112LL))(v10);
      return a2(a3, *(unsigned __int64 **)(v11 + 32), *(unsigned int *)(v11 + 40));
    }
  }
  else
  {
    v12[0] = *(_QWORD *)(a5 - 32);
    return a2(a3, v12, 1);
  }
}
