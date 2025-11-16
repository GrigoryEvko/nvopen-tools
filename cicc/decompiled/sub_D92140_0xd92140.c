// Function: sub_D92140
// Address: 0xd92140
//
char __fastcall sub_D92140(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rbx
  __int64 v5; // r14
  _QWORD *v6; // r15
  __int64 v7; // rax
  __int64 v8; // r14
  _QWORD *v9; // r14
  char result; // al
  _QWORD *v11; // rbx
  __int64 v12; // r13
  _QWORD *v13; // r15
  __int64 v14; // rax
  __int64 v15; // r13
  _QWORD *v16; // r13

  if ( *(_DWORD *)(a2 + 32) )
  {
    v11 = *(_QWORD **)(a1 + 40);
    v12 = 8LL * *(unsigned int *)(a1 + 48);
    v13 = &v11[(unsigned __int64)v12 / 8];
    v14 = v12 >> 3;
    v15 = v12 >> 5;
    if ( v15 )
    {
      v16 = &v11[4 * v15];
      while ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64, __int64))(*(_QWORD *)*v11 + 16LL))(*v11, a2, a3) )
      {
        if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64, __int64))(*(_QWORD *)v11[1] + 16LL))(v11[1], a2, a3) )
          return v13 != v11 + 1;
        if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64, __int64))(*(_QWORD *)v11[2] + 16LL))(v11[2], a2, a3) )
          return v13 != v11 + 2;
        if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64, __int64))(*(_QWORD *)v11[3] + 16LL))(v11[3], a2, a3) )
          return v13 != v11 + 3;
        v11 += 4;
        if ( v16 == v11 )
        {
          v14 = v13 - v11;
          goto LABEL_34;
        }
      }
      return v11 != v13;
    }
LABEL_34:
    if ( v14 != 2 )
    {
      if ( v14 != 3 )
      {
        if ( v14 != 1 )
          return 0;
        goto LABEL_45;
      }
      if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64, __int64))(*(_QWORD *)*v11 + 16LL))(*v11, a2, a3) )
        return v11 != v13;
      ++v11;
    }
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64, __int64))(*(_QWORD *)*v11 + 16LL))(*v11, a2, a3) )
      return v13 != v11;
    ++v11;
LABEL_45:
    result = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(*(_QWORD *)*v11 + 16LL))(*v11, a2, a3);
    if ( !result )
      return result;
    return v13 != v11;
  }
  v4 = *(_QWORD **)(a2 + 40);
  v5 = 8LL * *(unsigned int *)(a2 + 48);
  v6 = &v4[(unsigned __int64)v5 / 8];
  v7 = v5 >> 3;
  v8 = v5 >> 5;
  if ( v8 )
  {
    v9 = &v4[4 * v8];
    while ( (unsigned __int8)sub_D92140(a1, *v4, a3) )
    {
      if ( !(unsigned __int8)sub_D92140(a1, v4[1], a3) )
        return v6 == ++v4;
      if ( !(unsigned __int8)sub_D92140(a1, v4[2], a3) )
        return v6 == v4 + 2;
      if ( !(unsigned __int8)sub_D92140(a1, v4[3], a3) )
        return v6 == v4 + 3;
      v4 += 4;
      if ( v9 == v4 )
      {
        v7 = v6 - v4;
        goto LABEL_20;
      }
    }
    return v6 == v4;
  }
LABEL_20:
  if ( v7 == 2 )
    goto LABEL_29;
  if ( v7 == 3 )
  {
    if ( !(unsigned __int8)sub_D92140(a1, *v4, a3) )
      return v6 == v4;
    ++v4;
LABEL_29:
    if ( (unsigned __int8)sub_D92140(a1, *v4, a3) )
    {
      ++v4;
      goto LABEL_31;
    }
    return v6 == v4;
  }
  if ( v7 != 1 )
    return 1;
LABEL_31:
  result = sub_D92140(a1, *v4, a3);
  if ( !result )
    return v6 == v4;
  return result;
}
