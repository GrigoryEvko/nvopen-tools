// Function: sub_353DF70
// Address: 0x353df70
//
__int64 __fastcall sub_353DF70(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 *v4; // rcx
  __int64 v5; // rcx
  __int64 result; // rax

  v2 = a2 + ((*a1 - a1[1]) >> 3);
  if ( v2 < 0 )
  {
    v3 = ~((unsigned __int64)~v2 >> 6);
    goto LABEL_4;
  }
  if ( v2 > 63 )
  {
    v3 = v2 >> 6;
LABEL_4:
    v4 = (__int64 *)(a1[3] + 8 * v3);
    a1[3] = (__int64)v4;
    v5 = *v4;
    result = v5 + 8 * (v2 - (v3 << 6));
    a1[1] = v5;
    a1[2] = v5 + 512;
    *a1 = result;
    return result;
  }
  result = *a1 + 8 * a2;
  *a1 = result;
  return result;
}
