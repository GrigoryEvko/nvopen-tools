// Function: sub_2C46D80
// Address: 0x2c46d80
//
char __fastcall sub_2C46D80(__int64 a1)
{
  _QWORD *v1; // rbx
  __int64 v2; // r13
  _QWORD *v3; // r14
  __int64 v4; // rax
  __int64 v5; // r13
  _QWORD *v6; // r13
  char result; // al

  v1 = *(_QWORD **)(a1 + 16);
  v2 = 8LL * *(unsigned int *)(a1 + 24);
  v3 = &v1[(unsigned __int64)v2 / 8];
  v4 = v2 >> 3;
  v5 = v2 >> 5;
  if ( v5 )
  {
    v6 = &v1[4 * v5];
    while ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)*v1 + 32LL))(*v1, a1) )
    {
      if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v1[1] + 32LL))(v1[1], a1) )
        return v3 == v1 + 1;
      if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v1[2] + 32LL))(v1[2], a1) )
        return v3 == v1 + 2;
      if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v1[3] + 32LL))(v1[3], a1) )
        return v3 == v1 + 3;
      v1 += 4;
      if ( v1 == v6 )
      {
        v4 = v3 - v1;
        goto LABEL_11;
      }
    }
    return v3 == v1;
  }
LABEL_11:
  if ( v4 == 2 )
    goto LABEL_17;
  if ( v4 == 3 )
  {
    if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)*v1 + 32LL))(*v1, a1) )
      return v3 == v1;
    ++v1;
LABEL_17:
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)*v1 + 32LL))(*v1, a1) )
    {
      ++v1;
      goto LABEL_19;
    }
    return v3 == v1;
  }
  if ( v4 != 1 )
    return 1;
LABEL_19:
  result = (*(__int64 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)*v1 + 32LL))(*v1, a1);
  if ( !result )
    return v3 == v1;
  return result;
}
