// Function: sub_3176FF0
// Address: 0x3176ff0
//
__int64 __fastcall sub_3176FF0(__int64 **a1, _BYTE *a2)
{
  __int64 v2; // r12
  unsigned __int8 *v4; // rax

  if ( *a2 != 13
    && ((v2 = (__int64)a2, *a2 <= 0x15u) || (v2 = sub_2A66C60(*a1, (__int64)a2)) != 0)
    && (*(_BYTE *)(*(_QWORD *)(v2 + 8) + 8LL) != 14
     || sub_AC30F0(v2)
     || (v4 = sub_98ACB0((unsigned __int8 *)v2, 6u), *v4 != 3)
     || (v4[80] & 1) != 0
     || (_BYTE)qword_5034588) )
  {
    return v2;
  }
  else
  {
    return 0;
  }
}
