// Function: sub_1291800
// Address: 0x1291800
//
_QWORD *__fastcall sub_1291800(_QWORD *a1, __int64 a2, unsigned __int64 a3)
{
  char v5; // r14
  unsigned __int64 v7; // r13
  _BYTE *v8; // rsi
  _BYTE *v9; // rax
  _QWORD *v10; // r14
  _QWORD *i; // r15
  __int64 v12; // rdx
  unsigned __int64 v13; // [rsp+8h] [rbp-58h]
  __int64 v14; // [rsp+10h] [rbp-50h] BYREF
  _BYTE *v15; // [rsp+18h] [rbp-48h]
  _BYTE *v16; // [rsp+20h] [rbp-40h]

  v5 = *(_BYTE *)(a3 + 40);
  if ( sub_127C8B0(a3) )
  {
    v14 = 0;
    v15 = 0;
    v16 = 0;
    v7 = sub_127C950(a2 + 176, a3);
    if ( v5 == 6 )
    {
      v13 = sub_127C950(a2 + 176, *(_QWORD *)(*(_QWORD *)(a3 + 72) + 128LL));
      goto LABEL_7;
    }
  }
  else
  {
    if ( v5 != 8 )
    {
      *a1 = 0;
      a1[1] = 0;
      a1[2] = 0;
      return a1;
    }
    v14 = 0;
    v15 = 0;
    v16 = 0;
    v7 = sub_127C950(a2 + 176, a3);
  }
  v13 = *(_QWORD *)(a2 + 184);
LABEL_7:
  v8 = v15;
  v9 = v16;
  while ( v7 > v13 )
  {
    --v7;
    v10 = (_QWORD *)(*(_QWORD *)(a2 + 344) + 24 * v7);
    for ( i = (_QWORD *)v10[1]; (_QWORD *)*v10 != i; v15 = v8 )
    {
      while ( 1 )
      {
        --i;
        if ( v8 != v9 )
          break;
        sub_930F50((__int64)&v14, v8, i);
        v8 = v15;
        v9 = v16;
        if ( (_QWORD *)*v10 == i )
          goto LABEL_15;
      }
      if ( v8 )
      {
        *(_QWORD *)v8 = *i;
        v8 = v15;
        v9 = v16;
      }
      v8 += 8;
    }
LABEL_15:
    ;
  }
  v12 = v14;
  a1[1] = v8;
  *a1 = v12;
  a1[2] = v9;
  return a1;
}
