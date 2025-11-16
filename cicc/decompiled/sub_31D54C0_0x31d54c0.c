// Function: sub_31D54C0
// Address: 0x31d54c0
//
__int64 __fastcall sub_31D54C0(__int64 a1, unsigned __int64 a2, _BYTE *a3, unsigned int a4)
{
  __int64 *v6; // rdi
  __int64 v7; // rax
  bool v8; // zf
  void (*v9)(); // rax
  _BYTE *v11; // [rsp+0h] [rbp-50h] BYREF
  __int16 v12; // [rsp+20h] [rbp-30h]

  v6 = *(__int64 **)(a1 + 224);
  if ( a3 && *(_BYTE *)(a1 + 488) )
  {
    v7 = *v6;
    v8 = *a3 == 0;
    v12 = 257;
    v9 = *(void (**)())(v7 + 120);
    if ( !v8 )
    {
      v11 = a3;
      LOBYTE(v12) = 3;
    }
    if ( v9 != nullsub_98 )
    {
      ((void (__fastcall *)(__int64 *, _BYTE **, __int64))v9)(v6, &v11, 1);
      v6 = *(__int64 **)(a1 + 224);
    }
  }
  return sub_E98EB0((__int64)v6, a2, a4);
}
