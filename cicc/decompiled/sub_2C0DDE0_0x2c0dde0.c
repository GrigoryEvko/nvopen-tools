// Function: sub_2C0DDE0
// Address: 0x2c0dde0
//
_QWORD *__fastcall sub_2C0DDE0(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // r15
  __int64 v4; // r14
  __int64 v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rbx
  _QWORD *v8; // r13
  __int64 v9; // rdi
  unsigned __int8 (__fastcall *v10)(__int64, __int64); // rax
  __int64 v12; // rsi
  __int64 v13; // rsi
  _QWORD *v15; // [rsp+8h] [rbp-38h]

  v3 = a1;
  v4 = a3;
  v5 = a2 - (_QWORD)a1;
  v6 = v5 >> 3;
  if ( v5 >> 5 > 0 )
  {
    v7 = a3 + 96;
    v8 = &a1[4 * (v5 >> 5)];
    while ( 1 )
    {
      v9 = *v3;
      v10 = *(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)*v3 + 16LL);
      if ( v4 )
      {
        if ( !v10(v9, v7) )
          return v3;
        if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v3[1] + 16LL))(v3[1], v7) )
          return v3 + 1;
        if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v3[2] + 16LL))(v3[2], v7) )
          return v3 + 2;
        v15 = v3 + 3;
        if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v3[3] + 16LL))(v3[3], v7) )
          return v15;
      }
      else
      {
        if ( !v10(v9, 0) )
          return v3;
        if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, _QWORD))(*(_QWORD *)v3[1] + 16LL))(v3[1], 0) )
          return v3 + 1;
        if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, _QWORD))(*(_QWORD *)v3[2] + 16LL))(v3[2], 0) )
          return v3 + 2;
        v15 = v3 + 3;
        if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, _QWORD))(*(_QWORD *)v3[3] + 16LL))(v3[3], 0) )
          return v15;
      }
      v3 += 4;
      if ( v3 == v8 )
      {
        v6 = (a2 - (__int64)v3) >> 3;
        break;
      }
    }
  }
  if ( v6 == 2 )
  {
LABEL_24:
    v13 = v4 + 96;
    if ( !v4 )
      v13 = 0;
    v15 = v3;
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)*v3 + 16LL))(*v3, v13) )
    {
      ++v3;
LABEL_28:
      if ( v4 )
        v4 += 96;
      if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)*v3 + 16LL))(*v3, v4) )
        return (_QWORD *)a2;
      return v3;
    }
    return v15;
  }
  if ( v6 != 3 )
  {
    if ( v6 != 1 )
      return (_QWORD *)a2;
    goto LABEL_28;
  }
  v12 = v4 + 96;
  if ( !v4 )
    v12 = 0;
  v15 = v3;
  if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)*v3 + 16LL))(*v3, v12) )
  {
    ++v3;
    goto LABEL_24;
  }
  return v15;
}
