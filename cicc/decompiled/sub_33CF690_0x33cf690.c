// Function: sub_33CF690
// Address: 0x33cf690
//
__int64 __fastcall sub_33CF690(__int64 a1, unsigned __int64 a2)
{
  __int64 result; // rax
  __int64 *v4; // rax
  __int64 v5; // rcx

  for ( result = a1; *(_DWORD *)(result + 24) == 161; a2 = v5 | a2 & 0xFFFFFFFF00000000LL )
  {
    v4 = *(__int64 **)(result + 40);
    v5 = *((unsigned int *)v4 + 2);
    result = *v4;
  }
  return result;
}
