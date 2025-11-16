// Function: sub_1DA9A50
// Address: 0x1da9a50
//
_QWORD *__fastcall sub_1DA9A50(_QWORD *a1, __int64 *a2)
{
  _QWORD *v2; // r15
  _QWORD *v3; // r12
  __int64 v4; // r14
  unsigned int v5; // ecx
  unsigned int v6; // eax
  _QWORD *v7; // rdx
  _BOOL4 v8; // r8d
  __int64 v9; // r14
  __int64 v11; // rcx
  _BOOL4 v12; // [rsp+Ch] [rbp-34h]

  v2 = a1 + 1;
  v3 = (_QWORD *)a1[2];
  if ( !v3 )
  {
    v3 = a1 + 1;
    if ( v2 == (_QWORD *)a1[3] )
    {
      v8 = 1;
      goto LABEL_10;
    }
    v4 = *a2;
    goto LABEL_13;
  }
  v4 = *a2;
  v5 = *(_DWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*a2 >> 1) & 3;
  while ( 1 )
  {
    v6 = *(_DWORD *)((v3[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) | ((__int64)v3[4] >> 1) & 3;
    v7 = (_QWORD *)v3[3];
    if ( v5 < v6 )
      v7 = (_QWORD *)v3[2];
    if ( !v7 )
      break;
    v3 = v7;
  }
  if ( v5 >= v6 )
  {
    if ( v5 > v6 )
      goto LABEL_9;
    return v3;
  }
  if ( v3 != (_QWORD *)a1[3] )
  {
LABEL_13:
    v11 = sub_220EF80(v3);
    if ( (*(_DWORD *)((v4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v4 >> 1) & 3) <= (*(_DWORD *)((*(_QWORD *)(v11 + 32) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                         | (unsigned int)(*(__int64 *)(v11 + 32) >> 1)
                                                                                         & 3) )
      return (_QWORD *)v11;
    if ( v3 )
    {
      v8 = 1;
      if ( v2 == v3 )
        goto LABEL_10;
LABEL_18:
      v8 = (*(_DWORD *)((v4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v4 >> 1) & 3) < (*(_DWORD *)((v3[4] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                          | (unsigned int)((__int64)v3[4] >> 1)
                                                                                          & 3);
      goto LABEL_10;
    }
    return v3;
  }
LABEL_9:
  v8 = 1;
  if ( v2 != v3 )
    goto LABEL_18;
LABEL_10:
  v12 = v8;
  v9 = sub_22077B0(40);
  *(_QWORD *)(v9 + 32) = *a2;
  sub_220F040(v12, v9, v3, v2);
  ++a1[5];
  return (_QWORD *)v9;
}
