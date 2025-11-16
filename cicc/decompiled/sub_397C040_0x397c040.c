// Function: sub_397C040
// Address: 0x397c040
//
void __fastcall sub_397C040(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 *v4; // rdi
  __int64 v5; // rax
  bool v6; // zf
  void (*v7)(); // rax
  _BYTE *v8; // [rsp+0h] [rbp-30h] BYREF
  __int16 v9; // [rsp+10h] [rbp-20h]

  v4 = *(__int64 **)(a1 + 256);
  if ( a3 && *(_BYTE *)(a1 + 416) )
  {
    v5 = *v4;
    v6 = *a3 == 0;
    v9 = 257;
    v7 = *(void (**)())(v5 + 104);
    if ( !v6 )
    {
      v8 = a3;
      LOBYTE(v9) = 3;
    }
    if ( v7 != nullsub_580 )
    {
      ((void (__fastcall *)(__int64 *, _BYTE **, __int64))v7)(v4, &v8, 1);
      v4 = *(__int64 **)(a1 + 256);
    }
  }
  sub_38DCF20((__int64)v4, a2);
}
