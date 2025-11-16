// Function: sub_96AAC0
// Address: 0x96aac0
//
__int64 __fastcall sub_96AAC0(__int64 *a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 v5; // rax
  _QWORD *v6; // rax
  _QWORD *v7; // r13
  _QWORD *v8; // rax
  _QWORD *v9; // rbx

  result = sub_C33340();
  v3 = *a2;
  v4 = result;
  if ( *a1 == result )
  {
    if ( result == v3 )
    {
      if ( a2 != a1 )
      {
        v8 = (_QWORD *)a1[1];
        if ( v8 )
        {
          v9 = &v8[3 * *(v8 - 1)];
          if ( v8 != v9 )
          {
            do
            {
              v9 -= 3;
              if ( *v9 == v3 )
                sub_969EE0((__int64)v9);
              else
                sub_C338F0(v9);
            }
            while ( (_QWORD *)a1[1] != v9 );
          }
          j_j_j___libc_free_0_0(v9 - 1);
        }
        return sub_C3C840(a1, a2);
      }
    }
    else if ( a1 != a2 )
    {
      v6 = (_QWORD *)a1[1];
      if ( !v6 )
        return sub_C338E0(a1, a2);
      v7 = &v6[3 * *(v6 - 1)];
      if ( v6 != v7 )
      {
        do
        {
          v7 -= 3;
          if ( v4 == *v7 )
            sub_969EE0((__int64)v7);
          else
            sub_C338F0(v7);
        }
        while ( (_QWORD *)a1[1] != v7 );
      }
      j_j_j___libc_free_0_0(v7 - 1);
      v5 = *a2;
LABEL_6:
      if ( v4 != v5 )
        return sub_C338E0(a1, a2);
      return sub_C3C840(a1, a2);
    }
  }
  else
  {
    if ( result != v3 )
      return sub_C33870(a1, a2);
    if ( a1 != a2 )
    {
      sub_C338F0(a1);
      v5 = *a2;
      goto LABEL_6;
    }
  }
  return result;
}
