// Function: sub_1358D80
// Address: 0x1358d80
//
__int64 __fastcall sub_1358D80(__int64 a1, __int64 a2, __int64 a3, const __m128i *a4)
{
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v6; // rbx
  __int64 v7; // r12

  v4 = a1 + 8;
  v5 = 0;
  v6 = *(_QWORD *)(a1 + 16);
  if ( a1 + 8 != v6 )
  {
    while ( 1 )
    {
      v7 = v6;
      v6 = *(_QWORD *)(v6 + 8);
      if ( *(_QWORD *)(v7 + 32) || !sub_13588D0(v7, a2, a3, a4, *(_QWORD **)a1) )
        goto LABEL_3;
      if ( v5 )
      {
        sub_1357740(v5, v7, a1);
        if ( v4 == v6 )
          return v5;
      }
      else
      {
        v5 = v7;
LABEL_3:
        if ( v4 == v6 )
          return v5;
      }
    }
  }
  return v5;
}
