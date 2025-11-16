// Function: sub_D90B10
// Address: 0xd90b10
//
_QWORD *__fastcall sub_D90B10(__int64 a1, unsigned __int64 *a2)
{
  _QWORD *v2; // rbx
  __int64 v3; // r13
  int v4; // eax
  _QWORD *v5; // rdx
  bool v6; // sf
  _QWORD *result; // rax
  __int64 v8; // rax

  v2 = *(_QWORD **)(a1 + 16);
  if ( v2 )
  {
    while ( 1 )
    {
      v3 = (__int64)(v2 + 4);
      v4 = sub_C49970((__int64)a2, v2 + 4);
      v5 = (_QWORD *)v2[3];
      if ( v4 < 0 )
        v5 = (_QWORD *)v2[2];
      if ( !v5 )
        break;
      v2 = v5;
    }
    if ( v4 >= 0 )
      goto LABEL_8;
  }
  else
  {
    v2 = (_QWORD *)(a1 + 8);
  }
  result = 0;
  if ( *(_QWORD **)(a1 + 24) == v2 )
    return result;
  v8 = sub_220EF80(v2);
  v3 = v8 + 32;
  v2 = (_QWORD *)v8;
LABEL_8:
  v6 = (int)sub_C49970(v3, a2) < 0;
  result = v2;
  if ( v6 )
    return 0;
  return result;
}
