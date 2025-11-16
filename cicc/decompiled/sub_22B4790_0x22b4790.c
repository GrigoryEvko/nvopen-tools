// Function: sub_22B4790
// Address: 0x22b4790
//
__int64 __fastcall sub_22B4790(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdi
  __int64 v3; // rax
  __int64 v4; // r12

  v2 = *(__int64 **)(a1 + 80);
  v3 = *v2;
  v2[10] += 168;
  v4 = (v3 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v2[1] >= (unsigned __int64)(v4 + 168) && v3 )
    *v2 = v4 + 168;
  else
    v4 = sub_9D1E70((__int64)v2, 168, 168, 3);
  sub_22AE750(v4, a2);
  return v4;
}
