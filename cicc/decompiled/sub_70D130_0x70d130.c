// Function: sub_70D130
// Address: 0x70d130
//
__int64 __fastcall sub_70D130(__int64 a1)
{
  __int64 v2; // r12
  char v3; // al
  int v4; // r13d
  unsigned int v5; // r9d
  __int64 v6; // r12
  __int64 v7; // rdi
  char i; // al
  __int64 v10; // r15
  int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 **v15; // rbx
  __int64 *v16; // rdi
  char j; // al
  unsigned int v18; // [rsp+Ch] [rbp-44h]
  _BYTE v19[4]; // [rsp+14h] [rbp-3Ch] BYREF
  _BYTE v20[4]; // [rsp+18h] [rbp-38h] BYREF
  _BYTE v21[52]; // [rsp+1Ch] [rbp-34h] BYREF

  v2 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 96LL) + 32LL);
  if ( v2 )
  {
    v3 = *(_BYTE *)(v2 + 80);
    v4 = 0;
    if ( v3 != 17 )
      goto LABEL_3;
    v2 = *(_QWORD *)(v2 + 88);
    if ( v2 )
    {
      v3 = *(_BYTE *)(v2 + 80);
      v4 = 1;
LABEL_3:
      v5 = 0;
      if ( v3 == 10 )
        goto LABEL_18;
      while ( v4 )
      {
        v2 = *(_QWORD *)(v2 + 8);
        if ( !v2 )
          break;
        if ( *(_BYTE *)(v2 + 80) == 10 )
        {
LABEL_18:
          v10 = *(_QWORD *)(v2 + 88);
          if ( (*(_BYTE *)(v10 + 193) & 0x10) == 0 )
          {
            v18 = v5;
            v11 = sub_5F04A0(v2, 0, (__int64)v20, (__int64)v19, (__int64)v21);
            v5 = v18;
            if ( v11 )
            {
              v5 = sub_8D7760(v10, 0, v12, v13, v14, v18);
              if ( !v5 )
                return v5;
              v5 = 1;
            }
          }
        }
      }
      if ( v5 )
        return v5;
    }
  }
  v6 = *(_QWORD *)(a1 + 160);
  if ( v6 )
  {
    while ( 1 )
    {
      if ( (*(_BYTE *)(v6 + 144) & 0x50) != 0x40 )
      {
        v7 = sub_8D4130(*(_QWORD *)(v6 + 120));
        for ( i = *(_BYTE *)(v7 + 140); i == 12; i = *(_BYTE *)(v7 + 140) )
          v7 = *(_QWORD *)(v7 + 160);
        if ( (unsigned __int8)(i - 9) <= 2u && !(unsigned int)sub_70D130(v7) )
          break;
      }
      v6 = *(_QWORD *)(v6 + 112);
      if ( !v6 )
        goto LABEL_22;
    }
  }
  else
  {
LABEL_22:
    v15 = **(__int64 ****)(a1 + 168);
    if ( !v15 )
      return 1;
    while ( 1 )
    {
      while ( 1 )
      {
        if ( ((_BYTE)v15[12] & 1) != 0 )
        {
          v16 = v15[5];
          for ( j = *((_BYTE *)v16 + 140); j == 12; j = *((_BYTE *)v16 + 140) )
            v16 = (__int64 *)v16[20];
          if ( (unsigned __int8)(j - 9) <= 2u )
            break;
        }
        v15 = (__int64 **)*v15;
        if ( !v15 )
          return 1;
      }
      if ( !(unsigned int)sub_70D130(v16) )
        break;
      v15 = (__int64 **)*v15;
      if ( !v15 )
        return 1;
    }
  }
  return 0;
}
