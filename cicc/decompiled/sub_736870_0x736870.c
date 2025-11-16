// Function: sub_736870
// Address: 0x736870
//
_BYTE *__fastcall sub_736870(__int64 a1, int a2)
{
  _BYTE *result; // rax
  __int64 v3; // rsi
  _BYTE *v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  _BYTE *v8; // rax
  __int64 v9; // rax
  __int64 v10; // [rsp+8h] [rbp-18h] BYREF

  result = sub_735B90(a2, a1, &v10);
  if ( !result )
    return result;
  v3 = v10;
  if ( *(_QWORD *)(v10 + 32) == a1 )
    return result;
  v4 = result;
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) <= 2u )
  {
    if ( *(_QWORD *)a1 )
    {
      v5 = *(_QWORD *)(*(_QWORD *)a1 + 96LL);
      if ( v5 )
      {
        v6 = *(_QWORD *)(v5 + 160);
        if ( v6 )
        {
          if ( *(_QWORD *)(v6 + 112) == a1 )
            goto LABEL_10;
        }
      }
    }
  }
  v7 = *((_QWORD *)v4 + 13);
  if ( a1 != v7 )
  {
    do
    {
      v6 = v7;
      v7 = *(_QWORD *)(v7 + 112);
    }
    while ( a1 != v7 );
LABEL_10:
    *(_QWORD *)(v6 + 112) = *(_QWORD *)(a1 + 112);
    goto LABEL_11;
  }
  v6 = 0;
  *((_QWORD *)v4 + 13) = *(_QWORD *)(a1 + 112);
LABEL_11:
  v8 = *(_BYTE **)(a1 + 112);
  if ( v8 && (unsigned __int8)(v8[140] - 9) <= 2u )
  {
    v9 = *(_QWORD *)(*(_QWORD *)v8 + 96LL);
    if ( v9 )
      *(_QWORD *)(v9 + 160) = v6;
  }
  result = *(_BYTE **)(v3 + 32);
  *((_QWORD *)result + 14) = a1;
  *(_QWORD *)(v3 + 32) = a1;
  *(_QWORD *)(a1 + 112) = 0;
  return result;
}
