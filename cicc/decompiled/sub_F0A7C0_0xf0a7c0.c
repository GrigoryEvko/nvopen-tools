// Function: sub_F0A7C0
// Address: 0xf0a7c0
//
__int64 __fastcall sub_F0A7C0(__int64 a1, int a2, const char **a3, __int64 a4, unsigned __int16 a5)
{
  __int64 v7; // rax
  __int64 v8; // r12

  v7 = sub_BD2DA0(80);
  v8 = v7;
  if ( v7 )
  {
    sub_B44260(v7, a1, 55, 0x8000000u, a4, a5);
    *(_DWORD *)(v8 + 72) = a2;
    sub_BD6B50((unsigned __int8 *)v8, a3);
    sub_BD2A10(v8, *(_DWORD *)(v8 + 72), 1);
  }
  return v8;
}
