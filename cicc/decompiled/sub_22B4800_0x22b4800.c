// Function: sub_22B4800
// Address: 0x22b4800
//
__int64 __fastcall sub_22B4800(__int64 a1, __int64 a2, char a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  __int64 *v7; // rdi
  __int64 v9; // rax
  __int64 v10; // r12

  v7 = *(__int64 **)(a1 + 80);
  v9 = *v7;
  v7[10] += 168;
  v10 = (v9 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v7[1] >= (unsigned __int64)(v10 + 168) && v9 )
    *v7 = v10 + 168;
  else
    v10 = sub_9D1E70((__int64)v7, 168, 168, 3);
  sub_22AF450(v10, a2, a3, a4, a5, a6);
  return v10;
}
