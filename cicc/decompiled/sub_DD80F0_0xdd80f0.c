// Function: sub_DD80F0
// Address: 0xdd80f0
//
__int64 *__fastcall sub_DD80F0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  unsigned int v4; // r8d

  if ( sub_D97040((__int64)a1, *(_QWORD *)(a2 + 8)) )
    return sub_DD65B0((__int64)a1, (unsigned __int8 *)a2, v2, v3, v4);
  else
    return sub_DA3860(a1, a2);
}
