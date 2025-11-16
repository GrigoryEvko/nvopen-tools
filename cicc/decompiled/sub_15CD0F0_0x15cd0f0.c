// Function: sub_15CD0F0
// Address: 0x15cd0f0
//
char __fastcall sub_15CD0F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r15
  __int64 v6; // rsi
  __int64 v7; // r14
  __int64 v8; // rax
  char result; // al
  __int64 i; // rax
  char v11; // [rsp+7h] [rbp-49h]
  __int64 v12; // [rsp+8h] [rbp-48h]
  __int64 v13[8]; // [rsp+10h] [rbp-40h] BYREF

  v5 = sub_1648700(a3);
  v11 = *(_BYTE *)(v5 + 16);
  if ( v11 != 77 )
  {
    v7 = *(_QWORD *)(v5 + 40);
    if ( sub_15CC510(a1, v7) )
      goto LABEL_5;
    return 1;
  }
  if ( (*(_BYTE *)(v5 + 23) & 0x40) != 0 )
    v6 = *(_QWORD *)(v5 - 8);
  else
    v6 = v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF);
  v7 = *(_QWORD *)(v6 + 0xFFFFFFFD55555558LL * (unsigned int)((a3 - v6) >> 3) + 24LL * *(unsigned int *)(v5 + 56) + 8);
  if ( !sub_15CC510(a1, v7) )
    return 1;
LABEL_5:
  v12 = *(_QWORD *)(a2 + 40);
  if ( !sub_15CC510(a1, v12) )
    return 0;
  if ( *(_BYTE *)(a2 + 16) == 29 )
  {
    v8 = *(_QWORD *)(a2 - 48);
    v13[0] = v12;
    v13[1] = v8;
    return sub_15CCFD0(a1, v13, a3);
  }
  else
  {
    if ( v7 != v12 )
      return sub_15CC8F0(a1, v12, v7);
    result = 1;
    if ( v11 != 77 )
    {
      for ( i = *(_QWORD *)(v7 + 48); ; i = *(_QWORD *)(i + 8) )
      {
        if ( i )
        {
          if ( a2 == i - 24 )
            return v5 != a2;
          if ( v5 == i - 24 )
            return 0;
        }
      }
    }
  }
  return result;
}
