// Function: sub_F8E860
// Address: 0xf8e860
//
_QWORD *__fastcall sub_F8E860(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // r13
  _QWORD *v3; // rbx
  _QWORD *v4; // r12
  unsigned __int64 *v5; // r14
  int v6; // eax
  __int64 v7; // rdx
  __int64 v8; // rcx

  v2 = (_QWORD *)(a1 + 8);
  v3 = *(_QWORD **)(a1 + 16);
  if ( v3 )
  {
    v4 = (_QWORD *)(a1 + 8);
    v5 = (unsigned __int64 *)(*a2 + 24LL);
    do
    {
      while ( 1 )
      {
        v6 = sub_C49970(v3[4] + 24LL, v5);
        v7 = v3[2];
        v8 = v3[3];
        if ( v6 < 0 )
          break;
        v4 = v3;
        v3 = (_QWORD *)v3[2];
        if ( !v7 )
          goto LABEL_6;
      }
      v3 = (_QWORD *)v3[3];
    }
    while ( v8 );
LABEL_6:
    if ( v2 != v4 && (int)sub_C49970((__int64)v5, (unsigned __int64 *)(v4[4] + 24LL)) >= 0 )
      return v4;
  }
  return v2;
}
