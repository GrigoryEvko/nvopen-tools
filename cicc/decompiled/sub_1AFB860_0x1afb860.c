// Function: sub_1AFB860
// Address: 0x1afb860
//
_QWORD *__fastcall sub_1AFB860(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 v3; // rax
  _QWORD *v4; // rcx
  __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rdx
  _QWORD *result; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  bool v13; // zf

  v2 = (a2 - (__int64)a1) >> 5;
  v3 = (a2 - (__int64)a1) >> 3;
  if ( v2 <= 0 )
  {
LABEL_13:
    if ( v3 != 2 )
    {
      if ( v3 != 3 )
      {
        if ( v3 != 1 )
          return (_QWORD *)a2;
LABEL_23:
        v12 = *(_QWORD *)(*a1 + 48LL);
        if ( v12 )
        {
          v13 = *(_BYTE *)(v12 - 8) == 77;
          result = a1;
          if ( !v13 )
            return (_QWORD *)a2;
          return result;
        }
LABEL_31:
        BUG();
      }
      v10 = *(_QWORD *)(*a1 + 48LL);
      if ( !v10 )
        goto LABEL_31;
      if ( *(_BYTE *)(v10 - 8) == 77 )
        return a1;
      ++a1;
    }
    v11 = *(_QWORD *)(*a1 + 48LL);
    if ( !v11 )
      goto LABEL_31;
    if ( *(_BYTE *)(v11 - 8) != 77 )
    {
      ++a1;
      goto LABEL_23;
    }
    return a1;
  }
  v4 = &a1[4 * v2];
  while ( 1 )
  {
    v5 = *(_QWORD *)(*a1 + 48LL);
    if ( !v5 )
      goto LABEL_31;
    if ( *(_BYTE *)(v5 - 8) == 77 )
      return a1;
    v6 = *(_QWORD *)(a1[1] + 48LL);
    if ( !v6 )
      goto LABEL_31;
    if ( *(_BYTE *)(v6 - 8) == 77 )
      return a1 + 1;
    v7 = *(_QWORD *)(a1[2] + 48LL);
    if ( !v7 )
      goto LABEL_31;
    if ( *(_BYTE *)(v7 - 8) == 77 )
      return a1 + 2;
    v8 = *(_QWORD *)(a1[3] + 48LL);
    if ( !v8 )
      goto LABEL_31;
    if ( *(_BYTE *)(v8 - 8) == 77 )
      return a1 + 3;
    a1 += 4;
    if ( v4 == a1 )
    {
      v3 = (a2 - (__int64)a1) >> 3;
      goto LABEL_13;
    }
  }
}
