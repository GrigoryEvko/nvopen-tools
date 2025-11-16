// Function: sub_121AB80
// Address: 0x121ab80
//
__int64 __fastcall sub_121AB80(__int64 a1)
{
  unsigned __int64 v1; // r13
  unsigned int v2; // r15d
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // rsi
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 *v11; // rax
  unsigned int v12; // [rsp+4h] [rbp-6Ch] BYREF
  __int64 *v13; // [rsp+8h] [rbp-68h] BYREF
  unsigned int *v14[4]; // [rsp+10h] [rbp-60h] BYREF
  char v15; // [rsp+30h] [rbp-40h]
  char v16; // [rsp+31h] [rbp-3Fh]

  v1 = *(_QWORD *)(a1 + 232);
  v12 = *(_DWORD *)(a1 + 280);
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 3, "expected '=' after name")
    || (unsigned __int8)sub_120AFE0(a1, 292, "expected 'type' after '='") )
  {
    return 1;
  }
  v4 = *(_QWORD *)(a1 + 968);
  v13 = 0;
  v5 = a1 + 960;
  if ( !v4 )
  {
    v6 = a1 + 960;
LABEL_12:
    v14[0] = &v12;
    v6 = sub_1216660((_QWORD *)(a1 + 952), v6, v14);
    goto LABEL_13;
  }
  v6 = a1 + 960;
  do
  {
    while ( 1 )
    {
      v7 = *(_QWORD *)(v4 + 16);
      v8 = *(_QWORD *)(v4 + 24);
      if ( *(_DWORD *)(v4 + 32) >= v12 )
        break;
      v4 = *(_QWORD *)(v4 + 24);
      if ( !v8 )
        goto LABEL_10;
    }
    v6 = v4;
    v4 = *(_QWORD *)(v4 + 16);
  }
  while ( v7 );
LABEL_10:
  if ( v6 == v5 || v12 < *(_DWORD *)(v6 + 32) )
    goto LABEL_12;
LABEL_13:
  v2 = sub_121A7A0(a1, v1, byte_3F871B3, 0, (unsigned __int64 *)(v6 + 40), &v13);
  if ( !(_BYTE)v2 )
  {
    if ( *((_BYTE *)v13 + 8) == 15 )
      return v2;
    v9 = *(_QWORD *)(a1 + 968);
    if ( v9 )
    {
      v10 = a1 + 960;
      do
      {
        if ( *(_DWORD *)(v9 + 32) < v12 )
        {
          v9 = *(_QWORD *)(v9 + 24);
        }
        else
        {
          v10 = v9;
          v9 = *(_QWORD *)(v9 + 16);
        }
      }
      while ( v9 );
      if ( v5 != v10 && v12 >= *(_DWORD *)(v10 + 32) )
        goto LABEL_24;
    }
    else
    {
      v10 = a1 + 960;
    }
    v14[0] = &v12;
    v10 = sub_1216660((_QWORD *)(a1 + 952), v10, v14);
LABEL_24:
    if ( !*(_QWORD *)(v10 + 40) )
    {
      v11 = v13;
      *(_QWORD *)(v10 + 48) = 0;
      *(_QWORD *)(v10 + 40) = v11;
      return v2;
    }
    v16 = 1;
    v14[0] = (unsigned int *)"non-struct types may not be recursive";
    v15 = 3;
    sub_11FD800(a1 + 176, v1, (__int64)v14, 1);
  }
  return 1;
}
