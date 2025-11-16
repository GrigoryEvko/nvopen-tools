// Function: sub_38B85E0
// Address: 0x38b85e0
//
__int64 __fastcall sub_38B85E0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // r15
  int v6; // r13d
  unsigned int v7; // ebx
  unsigned int v8; // r14d
  __int64 v9; // rax
  bool v10; // dl
  unsigned int v11; // r8d
  int v13; // eax
  int v14; // [rsp+Ch] [rbp-44h]
  __int64 v15; // [rsp+10h] [rbp-40h]
  __int64 v16; // [rsp+18h] [rbp-38h]

  v3 = a3 & 0xFFFFFFFFFFFFFFF8LL;
  v15 = a3 >> 2;
  v16 = (a3 >> 2) & 1;
  v4 = sub_157EBA0(a2);
  if ( !v4 )
    goto LABEL_21;
  v14 = sub_15F4D60(v4);
  v5 = sub_157EBA0(a2);
  v6 = v14 >> 2;
  if ( v14 >> 2 <= 0 )
  {
    v13 = v14;
    v7 = 0;
LABEL_17:
    if ( v13 != 2 )
    {
      if ( v13 != 3 )
      {
        if ( v13 != 1 )
          goto LABEL_21;
        goto LABEL_20;
      }
      v9 = sub_15F4DF0(v5, v7);
      if ( v3 == v9 )
        goto LABEL_9;
      ++v7;
    }
    v9 = sub_15F4DF0(v5, v7);
    if ( v3 == v9 )
      goto LABEL_9;
    ++v7;
LABEL_20:
    v9 = sub_15F4DF0(v5, v7);
    if ( v3 == v9 )
      goto LABEL_9;
LABEL_21:
    LODWORD(v9) = 0;
    v10 = 1;
LABEL_13:
    if ( v16 )
      goto LABEL_10;
LABEL_14:
    v11 = 0;
    if ( v10 )
      return v11;
    goto LABEL_10;
  }
  v7 = 0;
  while ( 1 )
  {
    v9 = sub_15F4DF0(v5, v7);
    if ( v3 == v9 )
      break;
    v8 = v7 + 1;
    v9 = sub_15F4DF0(v5, v7 + 1);
    if ( v3 == v9
      || (v8 = v7 + 2, v9 = sub_15F4DF0(v5, v7 + 2), v3 == v9)
      || (v8 = v7 + 3, v9 = sub_15F4DF0(v5, v7 + 3), v3 == v9) )
    {
      v10 = v14 == v8;
      LOBYTE(v9) = v14 != v8;
      goto LABEL_13;
    }
    v7 += 4;
    if ( !--v6 )
    {
      v13 = v14 - v7;
      goto LABEL_17;
    }
  }
LABEL_9:
  v10 = v14 == v7;
  LOBYTE(v9) = v14 != v7;
  if ( !v16 )
    goto LABEL_14;
LABEL_10:
  LOBYTE(v9) = v15 & v9;
  return (unsigned int)v9 ^ 1;
}
