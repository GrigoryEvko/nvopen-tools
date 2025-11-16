// Function: sub_7DFC20
// Address: 0x7dfc20
//
__int64 __fastcall sub_7DFC20(__int64 a1)
{
  char v2; // al
  __int64 v3; // rax
  unsigned int v4; // r13d
  __int64 v6; // rbx
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdi
  _QWORD *v10; // rbx
  __int64 v11; // rbx
  __int64 i; // rax
  __int64 j; // rdi
  char v14; // [rsp+7h] [rbp-29h] BYREF
  _QWORD v15[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *(_BYTE *)(a1 + 28);
  if ( v2 == 17 )
  {
    v11 = *(_QWORD *)(a1 + 32);
    for ( i = *(_QWORD *)(v11 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    for ( j = *(_QWORD *)(i + 160); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    if ( (*(_BYTE *)(j + 89) & 1) != 0 && v11 == sub_72B7D0(j) )
      return 1;
    goto LABEL_10;
  }
  if ( v2 == 2 || v2 == 15 )
  {
LABEL_10:
    if ( !unk_4D04440 )
    {
      v6 = *(_QWORD *)(a1 + 112);
      if ( v6 )
      {
        while ( 1 )
        {
          while ( 1 )
          {
            sub_72F9F0(v6, a1, &v14, v15);
            if ( v14 == 2 )
            {
              if ( *(_QWORD *)(*(_QWORD *)v15[0] + 16LL) )
                break;
            }
            v6 = *(_QWORD *)(v6 + 112);
            if ( !v6 )
              goto LABEL_18;
          }
          if ( (unsigned int)sub_8D3410(*(_QWORD *)(v6 + 120)) )
            return 1;
          v6 = *(_QWORD *)(v6 + 112);
          if ( !v6 )
            goto LABEL_18;
        }
      }
      goto LABEL_18;
    }
    return 1;
  }
  v3 = *(_QWORD *)(a1 + 144);
  if ( v3 )
  {
    while ( !*(_DWORD *)(v3 + 160) )
    {
      v3 = *(_QWORD *)(v3 + 112);
      if ( !v3 )
        goto LABEL_18;
    }
    return 1;
  }
LABEL_18:
  v7 = *(_QWORD *)(a1 + 104);
  if ( v7 )
  {
    while ( 1 )
    {
      if ( (unsigned __int8)(*(_BYTE *)(v7 + 140) - 9) <= 2u && !(unsigned int)sub_736DD0(v7) )
      {
        v8 = *(_QWORD *)(v7 + 168);
        if ( (*(_BYTE *)(v8 + 109) & 0x20) != 0 )
          break;
        v9 = *(_QWORD *)(v8 + 152);
        if ( v9 )
        {
          if ( (*(_BYTE *)(v9 + 29) & 0x20) == 0 && (unsigned int)((__int64 (*)(void))sub_7DFC20)() )
            break;
        }
      }
      v7 = *(_QWORD *)(v7 + 112);
      if ( !v7 )
        goto LABEL_28;
    }
    v4 = 1;
  }
  else
  {
LABEL_28:
    v4 = 0;
  }
  v10 = *(_QWORD **)(a1 + 160);
  if ( v10 )
  {
    while ( !(unsigned int)sub_7DFC20(v10) )
    {
      v10 = (_QWORD *)*v10;
      if ( !v10 )
        return v4;
    }
    return 1;
  }
  return v4;
}
