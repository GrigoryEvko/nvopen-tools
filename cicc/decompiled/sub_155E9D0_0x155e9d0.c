// Function: sub_155E9D0
// Address: 0x155e9d0
//
__int64 __fastcall sub_155E9D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r12
  __int64 i; // r14
  __int64 v6; // r13
  __int64 *v7; // r15
  __int64 v8; // r12
  __int64 v10; // rax
  __int64 v12; // [rsp+8h] [rbp-48h]

  v4 = (a3 - 1) / 2;
  v12 = a3 & 1;
  if ( a2 >= v4 )
  {
    v7 = (__int64 *)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    v6 = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = v6 )
  {
    v6 = 2 * (i + 1);
    v7 = (__int64 *)(a1 + 16 * (i + 1));
    if ( sub_155E9A0(v7, *(v7 - 1)) )
    {
      --v6;
      v7 = (__int64 *)(a1 + 8 * v6);
    }
    *(_QWORD *)(a1 + 8 * i) = *v7;
    if ( v6 >= v4 )
      break;
  }
  if ( !v12 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == v6 )
    {
      v10 = *(_QWORD *)(a1 + 8 * (2 * v6 + 2) - 8);
      v6 = 2 * v6 + 1;
      *v7 = v10;
      v7 = (__int64 *)(a1 + 8 * v6);
    }
  }
  v8 = (v6 - 1) / 2;
  if ( v6 > a2 )
  {
    while ( 1 )
    {
      v7 = (__int64 *)(a1 + 8 * v6);
      if ( !sub_155E9A0((__int64 *)(a1 + 8 * v8), a4) )
        break;
      v6 = v8;
      *v7 = *(_QWORD *)(a1 + 8 * v8);
      if ( a2 >= v8 )
      {
        v7 = (__int64 *)(a1 + 8 * v8);
        break;
      }
      v8 = (v8 - 1) / 2;
    }
  }
LABEL_13:
  *v7 = a4;
  return a4;
}
