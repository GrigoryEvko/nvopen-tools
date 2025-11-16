// Function: sub_2E32880
// Address: 0x2e32880
//
__int64 *__fastcall sub_2E32880(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rbx
  int v5; // eax
  __int64 v6; // rsi
  __int64 i; // rbx
  int v8; // eax
  __int64 v9; // r15
  __int64 v10; // rax
  _QWORD *v11; // rax
  unsigned __int8 *v12; // rsi
  _QWORD v14[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = a2 + 48;
  *a1 = 0;
  v3 = sub_2E313E0(a2);
  if ( v3 != a2 + 48 )
  {
    v4 = v3;
    while ( 1 )
    {
      v5 = *(_DWORD *)(v4 + 44);
      if ( (v5 & 4) == 0 && (v5 & 8) != 0 )
      {
        if ( (unsigned __int8)sub_2E88A90(v4, 1024, 1) )
          break;
        goto LABEL_5;
      }
      if ( (*(_QWORD *)(*(_QWORD *)(v4 + 16) + 24LL) & 0x400LL) != 0 )
        break;
LABEL_5:
      if ( (*(_BYTE *)v4 & 4) != 0 )
      {
        v4 = *(_QWORD *)(v4 + 8);
        if ( v4 == v2 )
          return a1;
      }
      else
      {
        while ( (*(_BYTE *)(v4 + 44) & 8) != 0 )
          v4 = *(_QWORD *)(v4 + 8);
        v4 = *(_QWORD *)(v4 + 8);
        if ( v4 == v2 )
          return a1;
      }
    }
    if ( a1 != (__int64 *)(v4 + 56) )
    {
      if ( *a1 )
        sub_B91220((__int64)a1, *a1);
      v6 = *(_QWORD *)(v4 + 56);
      *a1 = v6;
      if ( v6 )
        sub_B96E90((__int64)a1, v6, 1);
    }
    if ( (*(_BYTE *)v4 & 4) == 0 )
    {
      while ( (*(_BYTE *)(v4 + 44) & 8) != 0 )
        v4 = *(_QWORD *)(v4 + 8);
    }
    for ( i = *(_QWORD *)(v4 + 8); i != v2; i = *(_QWORD *)(i + 8) )
    {
      v8 = *(_DWORD *)(i + 44);
      if ( (v8 & 4) != 0 || (v8 & 8) == 0 )
      {
        if ( (*(_QWORD *)(*(_QWORD *)(i + 16) + 24LL) & 0x400LL) != 0 )
        {
LABEL_23:
          v9 = sub_B10CD0(i + 56);
          v10 = sub_B10CD0((__int64)a1);
          v11 = sub_B026B0(v10, v9);
          sub_B10CB0(v14, (__int64)v11);
          if ( *a1 )
            sub_B91220((__int64)a1, *a1);
          v12 = (unsigned __int8 *)v14[0];
          *a1 = v14[0];
          if ( v12 )
            sub_B976B0((__int64)v14, v12, (__int64)a1);
        }
      }
      else if ( (unsigned __int8)sub_2E88A90(i, 1024, 1) )
      {
        goto LABEL_23;
      }
      if ( (*(_BYTE *)i & 4) == 0 )
      {
        while ( (*(_BYTE *)(i + 44) & 8) != 0 )
          i = *(_QWORD *)(i + 8);
      }
    }
  }
  return a1;
}
