// Function: sub_740760
// Address: 0x740760
//
_QWORD *__fastcall sub_740760(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  char v3; // al
  __int64 v4; // rdx
  _QWORD *v5; // r12
  _QWORD *v6; // rax
  _QWORD *v7; // r13
  _QWORD *v8; // rax
  _QWORD *v10; // r13
  _QWORD *v11; // rax

  v2 = a1;
  v3 = *(_BYTE *)(a1 + 173);
LABEL_2:
  if ( v3 == 9 )
  {
LABEL_10:
    v5 = *(_QWORD **)(v2 + 176);
    if ( *((_BYTE *)v5 + 48) != 3 )
    {
      v6 = sub_726700(5);
      v6[7] = v5;
      v7 = v6;
      *v6 = sub_73D720(*(const __m128i **)(v2 + 128));
      v8 = sub_725A70(3u);
      v8[7] = v7;
      *(_QWORD *)(v2 + 176) = v8;
      return v8 + 7;
    }
  }
  else
  {
    while ( v3 == 10 )
    {
      v4 = *(_QWORD *)(v2 + 176);
      if ( !v4 )
        goto LABEL_14;
      v3 = *(_BYTE *)(v4 + 173);
      v2 = *(_QWORD *)(v2 + 176);
      if ( v3 == 13 )
      {
        v2 = *(_QWORD *)(v4 + 120);
        v3 = *(_BYTE *)(v2 + 173);
      }
      if ( v3 != 11 )
        goto LABEL_2;
      do
      {
        v2 = *(_QWORD *)(v2 + 120);
        v3 = *(_BYTE *)(v2 + 173);
      }
      while ( v3 == 11 );
      if ( v3 == 9 )
        goto LABEL_10;
    }
    if ( v3 == 2 )
    {
      sub_740640(v2);
      v2 = *(_QWORD *)(v2 + 176);
    }
LABEL_14:
    v10 = sub_73A720((const __m128i *)v2, a2);
    v11 = sub_725A70(3u);
    v11[7] = v10;
    v5 = v11;
    sub_724A80(v2, 9);
    *(_QWORD *)(v2 + 176) = v5;
  }
  return v5 + 7;
}
