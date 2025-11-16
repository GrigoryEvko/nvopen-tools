// Function: sub_D90DB0
// Address: 0xd90db0
//
char __fastcall sub_D90DB0(__int64 a1)
{
  _QWORD *v1; // rbx
  __int64 v2; // r12
  _QWORD *v3; // r13
  __int64 v4; // rax
  __int64 v5; // r12
  _QWORD *v6; // r12
  char result; // al

  v1 = *(_QWORD **)(a1 + 40);
  v2 = 8LL * *(unsigned int *)(a1 + 48);
  v3 = &v1[(unsigned __int64)v2 / 8];
  v4 = v2 >> 3;
  v5 = v2 >> 5;
  if ( v5 )
  {
    v6 = &v1[4 * v5];
    while ( (*(unsigned __int8 (__fastcall **)(_QWORD))(*(_QWORD *)*v1 + 8LL))(*v1) )
    {
      if ( !(*(unsigned __int8 (__fastcall **)(_QWORD))(*(_QWORD *)v1[1] + 8LL))(v1[1]) )
        return v3 == v1 + 1;
      if ( !(*(unsigned __int8 (__fastcall **)(_QWORD))(*(_QWORD *)v1[2] + 8LL))(v1[2]) )
        return v3 == v1 + 2;
      if ( !(*(unsigned __int8 (__fastcall **)(_QWORD))(*(_QWORD *)v1[3] + 8LL))(v1[3]) )
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
  if ( v4 != 2 )
  {
    if ( v4 != 3 )
    {
      if ( v4 != 1 )
        return 1;
      goto LABEL_19;
    }
    if ( !(*(unsigned __int8 (__fastcall **)(_QWORD))(*(_QWORD *)*v1 + 8LL))(*v1) )
      return v1 == v3;
    ++v1;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(_QWORD))(*(_QWORD *)*v1 + 8LL))(*v1) )
    return v3 == v1;
  ++v1;
LABEL_19:
  result = (*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)*v1 + 8LL))(*v1);
  if ( !result )
    return v3 == v1;
  return result;
}
