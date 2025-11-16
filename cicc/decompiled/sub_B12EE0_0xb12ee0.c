// Function: sub_B12EE0
// Address: 0xb12ee0
//
char __fastcall sub_B12EE0(__int64 a1)
{
  __int64 v1; // rbp
  __int64 v3; // rdx
  unsigned __int8 **v4; // rax
  unsigned __int8 **v5; // rax
  __int64 v6; // rax
  __int64 v7; // [rsp-28h] [rbp-28h] BYREF
  unsigned __int8 **v8; // [rsp-20h] [rbp-20h]
  __int64 v9; // [rsp-8h] [rbp-8h]

  if ( (unsigned __int8)(**(_BYTE **)(a1 + 40) - 5) <= 0x1Fu )
    return 1;
  v9 = v1;
  if ( !(unsigned int)sub_B12A30(a1) )
  {
    v6 = sub_B11F60(a1 + 80);
    if ( !(unsigned __int8)sub_AF4500(v6) )
      return 1;
  }
  sub_B129C0(&v7, a1);
  v3 = v7;
  v4 = v8;
  if ( v8 == (unsigned __int8 **)v7 )
    return v8 != v4;
  while ( 1 )
  {
    v5 = (unsigned __int8 **)(v3 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v3 & 4) != 0 )
    {
      if ( (unsigned int)**((unsigned __int8 **)*v5 + 17) - 12 <= 1 )
        break;
LABEL_16:
      v3 = (unsigned __int64)(v5 + 1) | 4;
      v4 = (unsigned __int8 **)v3;
      goto LABEL_14;
    }
    if ( (unsigned int)*v5[17] - 12 <= 1 )
      break;
    if ( !v5 )
      goto LABEL_16;
    v4 = v5 + 18;
    v3 = (__int64)v4;
LABEL_14:
    if ( v8 == v4 )
      return v8 != v4;
  }
  v4 = (unsigned __int8 **)v3;
  return v8 != v4;
}
