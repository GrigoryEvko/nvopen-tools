// Function: sub_31D5440
// Address: 0x31d5440
//
__int64 __fastcall sub_31D5440(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 *v4; // rdi
  __int64 v5; // rax
  bool v6; // zf
  void (*v7)(); // rax
  _BYTE *v9; // [rsp+0h] [rbp-40h] BYREF
  __int16 v10; // [rsp+20h] [rbp-20h]

  v4 = *(__int64 **)(a1 + 224);
  if ( a3 && *(_BYTE *)(a1 + 488) )
  {
    v5 = *v4;
    v6 = *a3 == 0;
    v10 = 257;
    v7 = *(void (**)())(v5 + 120);
    if ( !v6 )
    {
      v9 = a3;
      LOBYTE(v10) = 3;
    }
    if ( v7 != nullsub_98 )
    {
      ((void (__fastcall *)(__int64 *, _BYTE **, __int64))v7)(v4, &v9, 1);
      v4 = *(__int64 **)(a1 + 224);
    }
  }
  return sub_E990E0((__int64)v4, a2);
}
