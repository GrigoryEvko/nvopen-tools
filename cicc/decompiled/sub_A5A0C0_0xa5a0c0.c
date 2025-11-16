// Function: sub_A5A0C0
// Address: 0xa5a0c0
//
__int64 __fastcall sub_A5A0C0(__int64 a1)
{
  bool v1; // zf
  __int64 v2; // r12
  __int64 v3; // rbx
  __int64 v4; // rdx
  __int64 v5; // r12
  __int64 v6; // rsi
  __int64 v7; // r14
  __int64 v8; // r15
  __int64 v9; // r12
  __int64 result; // rax
  __int64 j; // r15
  __int64 v12; // rdx
  char v13; // al
  __int64 i; // [rsp+8h] [rbp-58h]
  char v15; // [rsp+1Fh] [rbp-41h] BYREF
  __int64 v16; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v17[7]; // [rsp+28h] [rbp-38h] BYREF

  v1 = *(_BYTE *)(a1 + 25) == 0;
  *(_DWORD *)(a1 + 176) = 0;
  if ( v1 )
  {
    sub_A59FE0(a1, *(_QWORD *)(a1 + 16));
    v2 = *(_QWORD *)(a1 + 16);
    if ( (*(_BYTE *)(v2 + 2) & 1) == 0 )
      goto LABEL_3;
  }
  else
  {
    v2 = *(_QWORD *)(a1 + 16);
    if ( (*(_BYTE *)(v2 + 2) & 1) == 0 )
    {
LABEL_3:
      v3 = *(_QWORD *)(v2 + 96);
      v4 = v3;
      goto LABEL_4;
    }
  }
  sub_B2C6D0(v2);
  v3 = *(_QWORD *)(v2 + 96);
  v2 = *(_QWORD *)(a1 + 16);
  if ( (*(_BYTE *)(v2 + 2) & 1) != 0 )
    sub_B2C6D0(*(_QWORD *)(a1 + 16));
  v4 = *(_QWORD *)(v2 + 96);
LABEL_4:
  v5 = v4 + 40LL * *(_QWORD *)(v2 + 104);
  while ( v3 != v5 )
  {
    while ( (*(_BYTE *)(v3 + 7) & 0x10) != 0 )
    {
      v3 += 40;
      if ( v3 == v5 )
        goto LABEL_9;
    }
    v6 = v3;
    v3 += 40;
    sub_A58D70(a1, v6);
  }
LABEL_9:
  v7 = 0x8000000000041LL;
  v8 = *(_QWORD *)(a1 + 16);
  v9 = *(_QWORD *)(v8 + 80);
  result = v8 + 72;
  for ( i = v8 + 72; i != v9; v9 = *(_QWORD *)(v9 + 8) )
  {
    if ( !v9 )
      BUG();
    if ( (*(_BYTE *)(v9 - 17) & 0x10) == 0 )
      result = (__int64)sub_A58D70(a1, v9 - 24);
    for ( j = *(_QWORD *)(v9 + 32); v9 + 24 != j; j = *(_QWORD *)(j + 8) )
    {
      if ( !j )
        BUG();
      if ( *(_BYTE *)(*(_QWORD *)(j - 16) + 8LL) != 7 && (*(_BYTE *)(j - 17) & 0x10) == 0 )
        sub_A58D70(a1, j - 24);
      result = (unsigned int)*(unsigned __int8 *)(j - 24) - 34;
      if ( (unsigned __int8)(*(_BYTE *)(j - 24) - 34) <= 0x33u )
      {
        if ( _bittest64(&v7, result) )
        {
          v17[0] = *(_QWORD *)(j + 48);
          result = sub_A74680(v17);
          if ( result )
            result = sub_A59250(a1, result);
        }
      }
    }
  }
  if ( *(_QWORD *)(a1 + 80) )
  {
    v12 = *(_QWORD *)(a1 + 16);
    v16 = a1;
    v13 = *(_BYTE *)(a1 + 25);
    v17[0] = v12;
    v15 = v13;
    result = (*(__int64 (__fastcall **)(__int64, __int64 *, _QWORD *, char *))(a1 + 88))(a1 + 64, &v16, v17, &v15);
  }
  *(_BYTE *)(a1 + 24) = 1;
  return result;
}
