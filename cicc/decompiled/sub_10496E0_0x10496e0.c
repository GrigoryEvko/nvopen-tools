// Function: sub_10496E0
// Address: 0x10496e0
//
__int64 __fastcall sub_10496E0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdi
  __int64 v4; // [rsp-8h] [rbp-8h]

  v2 = *(__int64 **)(a1 + 8);
  if ( v2 )
    return sub_FDD2C0(v2, a2, 0);
  *((_BYTE *)&v4 - 8) = 0;
  return *(&v4 - 2);
}
