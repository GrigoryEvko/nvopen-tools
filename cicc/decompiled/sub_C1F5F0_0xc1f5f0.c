// Function: sub_C1F5F0
// Address: 0xc1f5f0
//
__int64 __fastcall sub_C1F5F0(__int64 *a1, _QWORD *a2, _QWORD *a3, _QWORD *a4)
{
  __int64 v7; // rcx
  unsigned __int64 v8; // rdi
  unsigned __int64 i; // rax
  __int64 v10; // r13
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rsi
  unsigned int v17; // r15d
  unsigned __int64 v18; // r13
  unsigned __int64 v19; // rsi
  unsigned __int64 v20; // rdi
  _QWORD v23[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = *a1;
  v8 = a1[1];
  for ( i = v8; ; --i )
  {
    if ( !i )
    {
      v11 = -2;
      v10 = -1;
      goto LABEL_6;
    }
    v10 = i - 1;
    if ( *(_BYTE *)(v7 + i - 1) == 58 )
      break;
  }
  v11 = i - 2;
LABEL_6:
  v12 = v11;
  if ( v8 <= v11 )
    v12 = v8;
  while ( 1 )
  {
    if ( !v12 )
    {
      v14 = v10;
      goto LABEL_13;
    }
    v13 = v12 - 1;
    if ( *(_BYTE *)(v7 + v12 - 1) == 58 )
      break;
    --v12;
  }
  v14 = v11 - v13;
  if ( v8 > v13 )
    v8 = v12 - 1;
LABEL_13:
  *a2 = v7;
  a2[1] = v8;
  v15 = a1[1];
  if ( v15 <= v12 )
    v12 = a1[1];
  v16 = v15 - v12;
  if ( v16 > v14 )
    v16 = v14;
  v17 = sub_C93C90(v12 + *a1, v16, 10, v23);
  if ( (_BYTE)v17 )
  {
    return 0;
  }
  else
  {
    v18 = v10 + 1;
    v19 = 0;
    *a3 = v23[0];
    v20 = a1[1];
    if ( v18 <= v20 )
    {
      v19 = v20 - v18;
      v20 = v18;
    }
    if ( !(unsigned __int8)sub_C93C90(*a1 + v20, v19, 10, v23) )
    {
      v17 = 1;
      *a4 = v23[0];
    }
  }
  return v17;
}
