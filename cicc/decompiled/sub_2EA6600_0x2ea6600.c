// Function: sub_2EA6600
// Address: 0x2ea6600
//
__int64 *__fastcall sub_2EA6600(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  unsigned __int64 v5; // rdx
  __int64 v6; // rsi
  unsigned __int8 *v7; // rsi
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __int64 v12; // rsi
  _QWORD v13[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = sub_2EA49A0(a2);
  if ( v3 )
  {
    v4 = *(_QWORD *)(v3 + 16);
    if ( v4 )
    {
      v5 = *(_QWORD *)(v4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v5 == v4 + 48 )
        goto LABEL_19;
      if ( !v5 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v5 - 24) - 30 > 0xA )
LABEL_19:
        BUG();
      v6 = *(_QWORD *)(v5 + 24);
      v13[0] = v6;
      if ( v6 )
      {
        sub_B96E90((__int64)v13, v6, 1);
        v7 = (unsigned __int8 *)v13[0];
        if ( v13[0] )
        {
          *a1 = v13[0];
          sub_B976B0((__int64)v13, v7, (__int64)a1);
          return a1;
        }
      }
    }
  }
  v9 = **(_QWORD **)(a2 + 32);
  if ( v9 && (v10 = *(_QWORD *)(v9 + 16)) != 0 )
  {
    v11 = *(_QWORD *)(v10 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v11 == v10 + 48 )
      goto LABEL_21;
    if ( !v11 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v11 - 24) - 30 > 0xA )
LABEL_21:
      BUG();
    v12 = *(_QWORD *)(v11 + 24);
    *a1 = v12;
    if ( !v12 )
      return a1;
    sub_B96E90((__int64)a1, v12, 1);
    return a1;
  }
  else
  {
    *a1 = 0;
    return a1;
  }
}
