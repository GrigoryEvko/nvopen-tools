// Function: sub_7313A0
// Address: 0x7313a0
//
__int64 __fastcall sub_7313A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 (__fastcall *v7)(__int64); // [rsp-E8h] [rbp-E8h] BYREF
  int v8; // [rsp-88h] [rbp-88h]

  if ( (*(_BYTE *)(a1 + 25) & 1) != 0 )
  {
    sub_76C7C0(&v7, a2, a3, a4, a5, a6);
    v7 = sub_72A4E0;
    v8 = 1;
    return sub_76CDC0(a1);
  }
  return result;
}
