// Function: sub_5E4D30
// Address: 0x5e4d30
//
_QWORD *__fastcall sub_5E4D30(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // r12
  _QWORD *v6; // r12
  __int64 v7; // rdi

  v4 = *a2;
  if ( *a2 )
  {
    while ( *(_QWORD *)(v4 + 88) != *(_QWORD *)(a1 + 88) )
    {
      v4 = *(_QWORD *)(v4 + 8);
      if ( !v4 )
        goto LABEL_6;
    }
    return (_QWORD *)v4;
  }
  else
  {
LABEL_6:
    v6 = (_QWORD *)sub_87FA00(8, a1 + 48, *(unsigned int *)(a1 + 40));
    sub_877E20(v6, 0, *(_QWORD *)(a1 + 64));
    v6[11] = *(_QWORD *)(a1 + 88);
    v7 = *(_QWORD *)(a1 + 96);
    if ( v7 )
      v6[12] = sub_5E4D30(v7, a2, a3);
    else
      v6[12] = a3;
    v6[1] = *a2;
    *a2 = (__int64)v6;
    return v6;
  }
}
