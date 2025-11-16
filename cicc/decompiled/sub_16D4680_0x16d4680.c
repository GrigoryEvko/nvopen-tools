// Function: sub_16D4680
// Address: 0x16d4680
//
__int64 __fastcall sub_16D4680(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 result; // rax
  __int64 (***v6)(void); // rdi
  __int64 (*v7)(void); // rdx
  __int64 (***v8)(void); // [rsp+8h] [rbp-18h] BYREF

  if ( !*(_QWORD *)(a2 + 16) )
    sub_4263D6(a1, a2, a3);
  (*(void (__fastcall **)(__int64 (****)(void)))(a2 + 24))(&v8);
  *a3 = 1;
  result = (__int64)v8;
  v6 = *(__int64 (****)(void))(a1 + 8);
  *(_QWORD *)(a1 + 8) = v8;
  v8 = v6;
  if ( v6 )
  {
    v7 = **v6;
    if ( (char *)v7 == (char *)sub_16D4120 )
      return (*v6)[2]();
    else
      return v7();
  }
  return result;
}
