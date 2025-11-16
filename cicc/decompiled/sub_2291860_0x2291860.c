// Function: sub_2291860
// Address: 0x2291860
//
__int64 __fastcall sub_2291860(__int64 a1, char a2, unsigned int a3, __int64 a4, __int64 a5)
{
  _QWORD *v7; // rax
  char v8; // r8
  __int64 result; // rax
  _QWORD *v10; // rcx
  int v11; // eax

  *(_BYTE *)(a4 + 144LL * a3 + 136) = a2;
  v7 = sub_22916C0(a1, a4);
  if ( !v7 || (v8 = sub_228DFC0(a1, 0x26u, (__int64)v7, a5), result = 0, !v8) )
  {
    v10 = sub_2291790(a1, a4);
    result = 1;
    if ( v10 )
    {
      LOBYTE(v11) = sub_228DFC0(a1, 0x26u, a5, (__int64)v10);
      return v11 ^ 1u;
    }
  }
  return result;
}
