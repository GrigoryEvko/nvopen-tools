// Function: sub_31AF930
// Address: 0x31af930
//
__int64 __fastcall sub_31AF930(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r12
  __int64 i; // r14
  __int64 v6; // r13
  __int64 v7; // r15
  __int64 v8; // r12
  __int64 v9; // r14
  __int64 v11; // rax
  __int64 v13; // [rsp+8h] [rbp-48h]

  v4 = (a3 - 1) / 2;
  v13 = a3 & 1;
  if ( a2 >= v4 )
  {
    v7 = a1 + 8 * a2;
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    v6 = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = v6 )
  {
    v6 = 2 * (i + 1);
    v7 = a1 + 16 * (i + 1);
    if ( sub_B445A0(*(_QWORD *)(*(_QWORD *)v7 + 16LL), *(_QWORD *)(*(_QWORD *)(v7 - 8) + 16LL)) )
    {
      --v6;
      v7 = a1 + 8 * v6;
    }
    *(_QWORD *)(a1 + 8 * i) = *(_QWORD *)v7;
    if ( v6 >= v4 )
      break;
  }
  if ( !v13 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == v6 )
    {
      v11 = *(_QWORD *)(a1 + 8 * (2 * v6 + 2) - 8);
      v6 = 2 * v6 + 1;
      *(_QWORD *)v7 = v11;
      v7 = a1 + 8 * v6;
    }
  }
  v8 = (v6 - 1) / 2;
  if ( v6 > a2 )
  {
    while ( 1 )
    {
      v9 = a1 + 8 * v8;
      v7 = a1 + 8 * v6;
      if ( !sub_B445A0(*(_QWORD *)(*(_QWORD *)v9 + 16LL), *(_QWORD *)(a4 + 16)) )
        break;
      v6 = v8;
      *(_QWORD *)v7 = *(_QWORD *)v9;
      if ( a2 >= v8 )
      {
        v7 = a1 + 8 * v8;
        break;
      }
      v8 = (v8 - 1) / 2;
    }
  }
LABEL_13:
  *(_QWORD *)v7 = a4;
  return a4;
}
