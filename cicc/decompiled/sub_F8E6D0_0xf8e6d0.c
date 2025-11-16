// Function: sub_F8E6D0
// Address: 0xf8e6d0
//
_QWORD *__fastcall sub_F8E6D0(_QWORD *a1, _QWORD *a2)
{
  _QWORD *v2; // r15
  unsigned __int64 *v3; // r14
  unsigned __int64 *v4; // r13
  int v5; // eax
  _QWORD *v6; // rdx
  unsigned int v8; // r14d
  __int64 v9; // r13
  __int64 v10; // [rsp+8h] [rbp-48h]
  __int64 v11; // [rsp+10h] [rbp-40h]
  _QWORD *v12; // [rsp+18h] [rbp-38h]

  v2 = (_QWORD *)a1[2];
  v12 = a1 + 1;
  if ( v2 )
  {
    v11 = *a2;
    v3 = (unsigned __int64 *)(*a2 + 24LL);
    while ( 1 )
    {
      v4 = (unsigned __int64 *)(v2[4] + 24LL);
      v5 = sub_C49970((__int64)v3, v4);
      v6 = (_QWORD *)v2[3];
      if ( v5 < 0 )
        v6 = (_QWORD *)v2[2];
      if ( !v6 )
        break;
      v2 = v6;
    }
    if ( v5 >= 0 )
    {
      if ( (int)sub_C49970((__int64)v4, v3) >= 0 )
        return v2;
      goto LABEL_10;
    }
    if ( (_QWORD *)a1[3] == v2 )
    {
LABEL_10:
      v8 = 1;
      if ( v12 != v2 )
        v8 = (unsigned int)sub_C49970(v11 + 24, (unsigned __int64 *)(v2[4] + 24LL)) >> 31;
      goto LABEL_12;
    }
  }
  else
  {
    v2 = a1 + 1;
    if ( v12 == (_QWORD *)a1[3] )
    {
      v2 = a1 + 1;
      v8 = 1;
LABEL_12:
      v9 = sub_22077B0(40);
      *(_QWORD *)(v9 + 32) = *a2;
      sub_220F040(v8, v9, v2, v12);
      ++a1[5];
      return (_QWORD *)v9;
    }
    v11 = *a2;
  }
  v10 = sub_220EF80(v2);
  if ( (int)sub_C49970(*(_QWORD *)(v10 + 32) + 24LL, (unsigned __int64 *)(v11 + 24)) < 0 )
    goto LABEL_10;
  return (_QWORD *)v10;
}
