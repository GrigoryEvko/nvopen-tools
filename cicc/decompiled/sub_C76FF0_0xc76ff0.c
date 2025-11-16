// Function: sub_C76FF0
// Address: 0xc76ff0
//
__int16 __fastcall sub_C76FF0(__int64 *a1, __int64 a2)
{
  unsigned int v3; // r13d
  unsigned int v4; // r14d
  __int64 v5; // rdx
  __int16 result; // ax
  unsigned int v7; // r13d
  int v8; // eax
  bool v9; // cc
  const void **v10; // rsi
  int v11; // eax
  bool v12; // dl
  int v13; // eax
  __int64 v14; // [rsp+0h] [rbp-40h]
  int v15; // [rsp+Ch] [rbp-34h]
  int v16; // [rsp+Ch] [rbp-34h]

  v3 = *((_DWORD *)a1 + 2);
  if ( v3 > 0x40 )
    v15 = sub_C44630((__int64)a1);
  else
    v15 = sub_39FAC40(*a1);
  v4 = *((_DWORD *)a1 + 6);
  if ( v4 > 0x40 )
  {
    if ( v3 != v15 + (unsigned int)sub_C44630((__int64)(a1 + 2)) )
      goto LABEL_18;
  }
  else
  {
    v14 = a1[2];
    if ( v3 != v15 + (unsigned int)sub_39FAC40(v14) )
    {
      v5 = v14;
      goto LABEL_6;
    }
  }
  v7 = *(_DWORD *)(a2 + 8);
  if ( v7 <= 0x40 )
  {
    v13 = sub_39FAC40(*(_QWORD *)a2);
    v9 = *(_DWORD *)(a2 + 24) <= 0x40u;
    v10 = (const void **)(a2 + 16);
    v16 = v13;
    if ( v9 )
      goto LABEL_11;
  }
  else
  {
    v8 = sub_C44630(a2);
    v9 = *(_DWORD *)(a2 + 24) <= 0x40u;
    v10 = (const void **)(a2 + 16);
    v16 = v8;
    if ( v9 )
    {
LABEL_11:
      v11 = sub_39FAC40(*(_QWORD *)(a2 + 16));
      goto LABEL_12;
    }
  }
  v11 = sub_C44630((__int64)v10);
LABEL_12:
  if ( v16 + v11 == v7 )
  {
    if ( v4 <= 0x40 )
      v12 = a1[2] == *(_QWORD *)(a2 + 16);
    else
      v12 = sub_C43C50((__int64)(a1 + 2), v10);
    LOBYTE(result) = v12;
    HIBYTE(result) = 1;
    return result;
  }
  if ( v4 <= 0x40 )
  {
    v5 = a1[2];
LABEL_6:
    if ( (*(_QWORD *)a2 & v5) != 0 )
      return 256;
    goto LABEL_19;
  }
LABEL_18:
  if ( (unsigned __int8)sub_C446A0(a1 + 2, (__int64 *)a2) )
    return 256;
LABEL_19:
  if ( *(_DWORD *)(a2 + 24) <= 0x40u )
  {
    if ( (*a1 & *(_QWORD *)(a2 + 16)) != 0 )
      return 256;
  }
  else if ( (unsigned __int8)sub_C446A0((__int64 *)(a2 + 16), a1) )
  {
    return 256;
  }
  return 0;
}
