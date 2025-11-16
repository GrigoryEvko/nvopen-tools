// Function: sub_258D1F0
// Address: 0x258d1f0
//
_QWORD *__fastcall sub_258D1F0(__int64 a1, __int64 a2)
{
  unsigned __int8 *v2; // rax
  __int64 v3; // r12
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rcx
  __int64 v7; // r14
  unsigned __int64 v8; // r13
  __int64 v9; // rdx
  __int64 v10; // rcx
  _QWORD *result; // rax
  __int64 v12; // rdx

  v2 = sub_250CBE0((__int64 *)(a1 + 72), a2);
  if ( !v2
    || (v3 = (__int64)v2, sub_B2FC80((__int64)v2))
    || *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(v3 + 24) + 16LL) + 8LL) == 7 )
  {
LABEL_17:
    result = (_QWORD *)*(unsigned __int8 *)(a1 + 104);
    *(_BYTE *)(a1 + 105) = (_BYTE)result;
    return result;
  }
  if ( (*(_BYTE *)(v3 + 2) & 1) != 0 )
  {
    sub_B2C6D0(v3, a2, v4, v5);
    v6 = *(_QWORD *)(v3 + 96);
    v7 = v6 + 40LL * *(_QWORD *)(v3 + 104);
    if ( (*(_BYTE *)(v3 + 2) & 1) != 0 )
    {
      sub_B2C6D0(v3, a2, v12, v6);
      v6 = *(_QWORD *)(v3 + 96);
    }
  }
  else
  {
    v6 = *(_QWORD *)(v3 + 96);
    v7 = v6 + 40LL * *(_QWORD *)(v3 + 104);
  }
  v8 = v6;
  if ( v6 != v7 )
  {
    while ( !(unsigned __int8)sub_B2D750(v8) )
    {
      v8 += 40LL;
      if ( v7 == v8 )
        goto LABEL_11;
    }
    sub_258BA20(a1, a2, (_BYTE *)(a1 + 88), v8, 0, 3, v3);
    *(_QWORD *)(a1 + 360) = v8;
  }
LABEL_11:
  if ( (sub_B2FC80(v3) || (unsigned __int8)sub_B2FC00((_BYTE *)v3))
    && !(unsigned __int8)sub_B19060(*(_QWORD *)(a2 + 208) + 248LL, v3, v9, v10)
    && (!*(_QWORD *)(a2 + 4432) || !(*(unsigned __int8 (__fastcall **)(__int64, __int64))(a2 + 4440))(a2 + 4416, v3))
    || (result = sub_2566C40(a2 + 32, (__int64 *)(a1 + 72))) != 0 )
  {
    if ( *(_QWORD *)(a1 + 360) )
    {
      result = (_QWORD *)*(unsigned __int8 *)(a1 + 105);
      *(_BYTE *)(a1 + 104) = (_BYTE)result;
      return result;
    }
    goto LABEL_17;
  }
  return result;
}
