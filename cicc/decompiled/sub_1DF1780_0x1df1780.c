// Function: sub_1DF1780
// Address: 0x1df1780
//
__int64 __fastcall sub_1DF1780(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // r8
  _QWORD *v4; // rax
  __int64 v5; // rcx
  __int64 v6; // r9
  _QWORD *v7; // rcx

  v3 = a2[12];
  v4 = (_QWORD *)a2[11];
  v5 = (v3 - (__int64)v4) >> 5;
  v6 = (v3 - (__int64)v4) >> 3;
  if ( v5 > 0 )
  {
    v7 = &v4[4 * v5];
    while ( a3 != *v4 )
    {
      if ( a3 == v4[1] )
        return sub_1DF1770(a1, a2, (__int64)(v4 + 1));
      if ( a3 == v4[2] )
        return sub_1DF1770(a1, a2, (__int64)(v4 + 2));
      if ( a3 == v4[3] )
        return sub_1DF1770(a1, a2, (__int64)(v4 + 3));
      v4 += 4;
      if ( v4 == v7 )
      {
        v6 = (v3 - (__int64)v4) >> 3;
        goto LABEL_11;
      }
    }
    goto LABEL_8;
  }
LABEL_11:
  if ( v6 == 2 )
  {
LABEL_19:
    if ( a3 != *v4 )
    {
      ++v4;
      goto LABEL_14;
    }
    goto LABEL_8;
  }
  if ( v6 != 3 )
  {
    if ( v6 != 1 )
      return sub_1DF1770(a1, a2, v3);
LABEL_14:
    if ( a3 == *v4 )
      v3 = (__int64)v4;
    return sub_1DF1770(a1, a2, v3);
  }
  if ( a3 != *v4 )
  {
    ++v4;
    goto LABEL_19;
  }
LABEL_8:
  v3 = (__int64)v4;
  return sub_1DF1770(a1, a2, v3);
}
