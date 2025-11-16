// Function: sub_2647750
// Address: 0x2647750
//
__int64 __fastcall sub_2647750(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  _QWORD *v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  _QWORD *v6; // rdx
  __int64 v8; // rdi

  v2 = *(_QWORD *)(a1 + 80);
  v3 = *(_QWORD **)(a1 + 72);
  v4 = (v2 - (__int64)v3) >> 6;
  v5 = (v2 - (__int64)v3) >> 4;
  if ( v4 > 0 )
  {
    v6 = &v3[8 * v4];
    while ( a2 != *v3 )
    {
      if ( a2 == v3[2] )
        return sub_26476C0(a1 + 72, (__int64)(v3 + 2));
      if ( a2 == v3[4] )
        return sub_26476C0(a1 + 72, (__int64)(v3 + 4));
      if ( a2 == v3[6] )
        return sub_26476C0(a1 + 72, (__int64)(v3 + 6));
      v3 += 8;
      if ( v3 == v6 )
      {
        v5 = (v2 - (__int64)v3) >> 4;
        goto LABEL_11;
      }
    }
    goto LABEL_8;
  }
LABEL_11:
  if ( v5 == 2 )
    goto LABEL_19;
  if ( v5 == 3 )
  {
    if ( a2 == *v3 )
      goto LABEL_8;
    v3 += 2;
LABEL_19:
    if ( a2 != *v3 )
    {
      v3 += 2;
      goto LABEL_14;
    }
LABEL_8:
    v2 = (__int64)v3;
    return sub_26476C0(a1 + 72, v2);
  }
  if ( v5 != 1 )
    return sub_26476C0(a1 + 72, v2);
LABEL_14:
  v8 = a1 + 72;
  if ( a2 == *v3 )
    v2 = (__int64)v3;
  return sub_26476C0(v8, v2);
}
