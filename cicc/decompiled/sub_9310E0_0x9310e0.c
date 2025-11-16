// Function: sub_9310E0
// Address: 0x9310e0
//
__int64 __fastcall sub_9310E0(__int64 a1, __int64 a2, unsigned __int64 a3)
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
  if ( sub_91CC00(a3) )
  {
    v14 = 0;
    v15 = 0;
    v16 = 0;
    v7 = sub_91CCA0(a2 + 248, a3);
    if ( v5 == 6 )
    {
      v13 = sub_91CCA0(a2 + 248, *(_QWORD *)(*(_QWORD *)(a3 + 72) + 128LL));
      goto LABEL_7;
    }
  }
  else
  {
    if ( v5 != 8 )
    {
      *(_QWORD *)(a1 + 16) = 0;
      *(_OWORD *)a1 = 0;
      return a1;
    }
    v14 = 0;
    v15 = 0;
    v16 = 0;
    v7 = sub_91CCA0(a2 + 248, a3);
  }
  v13 = *(_QWORD *)(a2 + 256);
LABEL_7:
  v8 = v15;
  v9 = v16;
  while ( v7 > v13 )
  {
    --v7;
    v10 = (_QWORD *)(*(_QWORD *)(a2 + 416) + 24 * v7);
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
  *(_QWORD *)(a1 + 8) = v8;
  *(_QWORD *)a1 = v12;
  *(_QWORD *)(a1 + 16) = v9;
  return a1;
}
