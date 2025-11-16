// Function: sub_FD7AA0
// Address: 0xfd7aa0
//
__int64 __fastcall sub_FD7AA0(__int64 a1, const __m128i *a2, __int64 a3, _BYTE *a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // r13
  __int64 v8; // rbx
  __int64 v9; // r12
  char v10; // al

  v6 = 0;
  v7 = a1 + 8;
  *a4 = 1;
  v8 = *(_QWORD *)(a1 + 16);
  if ( v8 != a1 + 8 )
  {
    while ( 1 )
    {
      v9 = v8;
      v8 = *(_QWORD *)(v8 + 8);
      if ( *(_QWORD *)(v9 + 16) )
        goto LABEL_3;
      if ( a3 == v9 )
        goto LABEL_9;
      v10 = sub_FD5F50(v9, a2, *(__int64 **)a1);
      if ( !v10 )
      {
LABEL_3:
        if ( v7 == v8 )
          return v6;
      }
      else
      {
        if ( v10 != 3 )
          *a4 = 0;
LABEL_9:
        if ( !v6 )
        {
          v6 = v9;
          goto LABEL_3;
        }
        sub_FD7340(v6, v9, a1, *(__int64 **)a1, a5, a6);
        if ( v7 == v8 )
          return v6;
      }
    }
  }
  return v6;
}
