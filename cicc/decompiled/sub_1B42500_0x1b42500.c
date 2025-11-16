// Function: sub_1B42500
// Address: 0x1b42500
//
_QWORD *__fastcall sub_1B42500(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // r13
  unsigned __int64 *v3; // r15
  unsigned __int64 *v4; // r14
  int v5; // eax
  _QWORD *v6; // rdx
  bool v7; // sf
  _QWORD *result; // rax

  v2 = *(_QWORD **)(a1 + 16);
  if ( v2 )
  {
    v3 = (unsigned __int64 *)(*a2 + 24LL);
    while ( 1 )
    {
      v4 = (unsigned __int64 *)(v2[4] + 24LL);
      v5 = sub_16A9900((__int64)v3, v4);
      v6 = (_QWORD *)v2[3];
      if ( v5 < 0 )
        v6 = (_QWORD *)v2[2];
      if ( !v6 )
        break;
      v2 = v6;
    }
    if ( v5 >= 0 )
      goto LABEL_8;
  }
  else
  {
    v2 = (_QWORD *)(a1 + 8);
  }
  result = 0;
  if ( *(_QWORD **)(a1 + 24) == v2 )
    return result;
  v2 = (_QWORD *)sub_220EF80(v2);
  v3 = (unsigned __int64 *)(*a2 + 24LL);
  v4 = (unsigned __int64 *)(v2[4] + 24LL);
LABEL_8:
  v7 = (int)sub_16A9900((__int64)v4, v3) < 0;
  result = v2;
  if ( v7 )
    return 0;
  return result;
}
