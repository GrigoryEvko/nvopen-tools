// Function: sub_138A920
// Address: 0x138a920
//
__int64 __fastcall sub_138A920(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 v7; // r13
  __int64 v8; // rdx
  __int64 v9; // rbx
  __int64 result; // rax
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // [rsp+0h] [rbp-80h]
  unsigned __int64 v16; // [rsp+8h] [rbp-78h]
  __int64 v17; // [rsp+10h] [rbp-70h]
  _QWORD v18[12]; // [rsp+20h] [rbp-60h] BYREF

  v2 = sub_1632FA0(*(_QWORD *)(a2 + 40));
  v3 = *a1;
  v18[1] = v2;
  v4 = a1[1];
  v18[0] = v3;
  v18[2] = v4;
  v18[4] = a1 + 6;
  v5 = *(_QWORD *)(a2 + 80);
  v15 = (__int64)(a1 + 2);
  v18[3] = a1 + 2;
  if ( a2 + 72 != v5 )
  {
    while ( 1 )
    {
      if ( !v5 )
        BUG();
      v6 = *(_QWORD *)(v5 + 24);
      v7 = v5 + 16;
      if ( v5 + 16 != v6 )
        break;
LABEL_12:
      v5 = *(_QWORD *)(v5 + 8);
      if ( a2 + 72 == v5 )
        goto LABEL_13;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        if ( !v6 )
          BUG();
        v8 = *(unsigned __int8 *)(v6 - 8);
        if ( (unsigned int)(v8 - 25) <= 9 )
          break;
        if ( (unsigned __int8)(v8 - 75) > 1u && (_BYTE)v8 != 57 )
          goto LABEL_11;
LABEL_7:
        v6 = *(_QWORD *)(v6 + 8);
        if ( v7 == v6 )
          goto LABEL_12;
      }
      v8 = (unsigned int)v8 & 0xFFFFFFFB;
      if ( (_BYTE)v8 != 25 )
        goto LABEL_7;
LABEL_11:
      sub_138A5B0((__int64)v18, v6 - 24, v8);
      v6 = *(_QWORD *)(v6 + 8);
      if ( v7 == v6 )
        goto LABEL_12;
    }
  }
LABEL_13:
  if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
  {
    sub_15E08E0(a2);
    v9 = *(_QWORD *)(a2 + 88);
    result = 5LL * *(_QWORD *)(a2 + 96);
    v11 = v9 + 40LL * *(_QWORD *)(a2 + 96);
    if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
    {
      result = sub_15E08E0(a2);
      v9 = *(_QWORD *)(a2 + 88);
    }
  }
  else
  {
    v9 = *(_QWORD *)(a2 + 88);
    result = 5LL * *(_QWORD *)(a2 + 96);
    v11 = v9 + 40LL * *(_QWORD *)(a2 + 96);
  }
  for ( ; v11 != v9; result = sub_13848E0(v15, v14, 1u, v13) )
  {
    while ( 1 )
    {
      result = *(_QWORD *)v9;
      if ( *(_BYTE *)(*(_QWORD *)v9 + 8LL) == 15 )
        break;
      v9 += 40;
      if ( v11 == v9 )
        return result;
    }
    v12 = sub_14C81A0(v9);
    v17 &= 0xFFFFFFFF00000000LL;
    sub_13848E0(v15, v9, v17, v12);
    v13 = sub_14C8170();
    v14 = v9;
    v9 += 40;
    v16 = v16 & 0xFFFFFFFF00000000LL | 1;
  }
  return result;
}
