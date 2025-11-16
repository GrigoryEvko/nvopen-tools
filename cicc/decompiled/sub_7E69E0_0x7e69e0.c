// Function: sub_7E69E0
// Address: 0x7e69e0
//
_BYTE *__fastcall sub_7E69E0(_QWORD *a1, int *a2)
{
  _BYTE *v2; // r13

  if ( (unsigned int)(*a2 - 3) <= 2 )
  {
    sub_7E67B0(a1);
    sub_7E25D0((__int64)a1, a2);
    return 0;
  }
  else
  {
    v2 = sub_732B10((__int64)a1);
    sub_7E6810((__int64)v2, (__int64)a2, 1);
    return v2;
  }
}
