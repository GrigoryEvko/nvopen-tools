// Function: sub_2E441D0
// Address: 0x2e441d0
//
__int64 __fastcall sub_2E441D0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // rax
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rcx
  _QWORD *v8; // rcx

  v3 = *(_QWORD **)(a2 + 112);
  v4 = 8LL * *(unsigned int *)(a2 + 120);
  v5 = (__int64)&v3[(unsigned __int64)v4 / 8];
  v6 = v4 >> 3;
  v7 = v4 >> 5;
  if ( v7 )
  {
    v8 = &v3[4 * v7];
    while ( a3 != *v3 )
    {
      if ( a3 == v3[1] )
        return sub_2E441C0(a1, a2, (__int64)(v3 + 1));
      if ( a3 == v3[2] )
        return sub_2E441C0(a1, a2, (__int64)(v3 + 2));
      if ( a3 == v3[3] )
        return sub_2E441C0(a1, a2, (__int64)(v3 + 3));
      v3 += 4;
      if ( v8 == v3 )
      {
        v6 = (v5 - (__int64)v3) >> 3;
        goto LABEL_11;
      }
    }
    goto LABEL_8;
  }
LABEL_11:
  if ( v6 == 2 )
  {
LABEL_19:
    if ( a3 != *v3 )
    {
      ++v3;
      goto LABEL_14;
    }
    goto LABEL_8;
  }
  if ( v6 != 3 )
  {
    if ( v6 != 1 )
      return sub_2E441C0(a1, a2, v5);
LABEL_14:
    if ( a3 == *v3 )
      v5 = (__int64)v3;
    return sub_2E441C0(a1, a2, v5);
  }
  if ( a3 != *v3 )
  {
    ++v3;
    goto LABEL_19;
  }
LABEL_8:
  v5 = (__int64)v3;
  return sub_2E441C0(a1, a2, v5);
}
