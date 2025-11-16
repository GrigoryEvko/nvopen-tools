// Function: sub_1AD4AC0
// Address: 0x1ad4ac0
//
__int64 __fastcall sub_1AD4AC0(_QWORD *a1)
{
  __int64 result; // rax
  unsigned __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rbx

  result = 0;
  v2 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(char *)(v2 + 23) < 0 )
  {
    v3 = sub_1648A40(*a1 & 0xFFFFFFFFFFFFFFF8LL);
    v5 = v3 + v4;
    if ( *(char *)(v2 + 23) >= 0 )
      return v5 >> 4;
    else
      return (unsigned int)((v5 - sub_1648A40(v2)) >> 4);
  }
  return result;
}
