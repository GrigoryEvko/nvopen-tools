// Function: sub_154F040
// Address: 0x154f040
//
__int64 __fastcall sub_154F040(__int64 a1)
{
  bool v1; // zf
  __int64 v2; // r12
  __int64 v3; // rbx
  __int64 v4; // rdx
  __int64 v5; // r12
  __int64 v6; // rsi
  __int64 v7; // r15
  __int64 v8; // r12
  __int64 result; // rax
  __int64 v10; // r15
  __int64 v11; // rbx
  unsigned __int8 v12; // cl
  unsigned __int64 v13; // rax
  __int64 v14; // [rsp+0h] [rbp-50h]
  __int64 v15[7]; // [rsp+18h] [rbp-38h] BYREF

  v1 = *(_BYTE *)(a1 + 17) == 0;
  *(_DWORD *)(a1 + 104) = 0;
  if ( v1 )
  {
    sub_154EA00(a1, *(_QWORD *)(a1 + 8));
    v2 = *(_QWORD *)(a1 + 8);
    if ( (*(_BYTE *)(v2 + 18) & 1) == 0 )
      goto LABEL_3;
  }
  else
  {
    v2 = *(_QWORD *)(a1 + 8);
    if ( (*(_BYTE *)(v2 + 18) & 1) == 0 )
    {
LABEL_3:
      v3 = *(_QWORD *)(v2 + 88);
      v4 = v3;
      goto LABEL_4;
    }
  }
  sub_15E08E0(v2);
  v3 = *(_QWORD *)(v2 + 88);
  v2 = *(_QWORD *)(a1 + 8);
  if ( (*(_BYTE *)(v2 + 18) & 1) != 0 )
    sub_15E08E0(*(_QWORD *)(a1 + 8));
  v4 = *(_QWORD *)(v2 + 88);
LABEL_4:
  v5 = v4 + 40LL * *(_QWORD *)(v2 + 96);
  while ( v3 != v5 )
  {
    while ( (*(_BYTE *)(v3 + 23) & 0x20) != 0 )
    {
      v3 += 40;
      if ( v3 == v5 )
        goto LABEL_9;
    }
    v6 = v3;
    v3 += 40;
    sub_154E370(a1, v6);
  }
LABEL_9:
  v7 = *(_QWORD *)(a1 + 8);
  v8 = *(_QWORD *)(v7 + 80);
  result = v7 + 72;
  v14 = v7 + 72;
  if ( v8 != v7 + 72 )
  {
    while ( 1 )
    {
      if ( !v8 )
        BUG();
      if ( (*(_BYTE *)(v8 - 1) & 0x20) == 0 )
        result = (__int64)sub_154E370(a1, v8 - 24);
      v10 = *(_QWORD *)(v8 + 24);
      v11 = v8 + 16;
      if ( v8 + 16 != v10 )
        break;
LABEL_27:
      v8 = *(_QWORD *)(v8 + 8);
      if ( v14 == v8 )
        goto LABEL_28;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        if ( !v10 )
          BUG();
        result = v10 - 24;
        if ( *(_BYTE *)(*(_QWORD *)(v10 - 24) + 8LL) && (*(_BYTE *)(v10 - 1) & 0x20) == 0 )
        {
          sub_154E370(a1, v10 - 24);
          result = v10 - 24;
        }
        v12 = *(_BYTE *)(v10 - 8);
        if ( v12 <= 0x17u )
          goto LABEL_16;
        if ( v12 != 78 )
          break;
        v13 = result | 4;
LABEL_24:
        result = v13 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !result )
          goto LABEL_16;
        v15[0] = *(_QWORD *)(result + 56);
        result = sub_1560250(v15);
        if ( !result )
          goto LABEL_16;
        result = (__int64)sub_154EC60(a1, result);
        v10 = *(_QWORD *)(v10 + 8);
        if ( v11 == v10 )
          goto LABEL_27;
      }
      if ( v12 == 29 )
      {
        v13 = result & 0xFFFFFFFFFFFFFFFBLL;
        goto LABEL_24;
      }
LABEL_16:
      v10 = *(_QWORD *)(v10 + 8);
      if ( v11 == v10 )
        goto LABEL_27;
    }
  }
LABEL_28:
  *(_BYTE *)(a1 + 16) = 1;
  return result;
}
