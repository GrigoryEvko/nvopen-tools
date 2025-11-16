// Function: sub_70D330
// Address: 0x70d330
//
__int64 __fastcall sub_70D330(__int64 a1)
{
  __int64 v2; // r12
  char v3; // al
  int v4; // r13d
  unsigned int v5; // r15d
  __int64 v6; // r12
  __int64 v7; // rdi
  char j; // al
  __int64 v10; // r14
  __int64 i; // rdi
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 **v16; // rbx
  __int64 *v17; // rdi
  char k; // al

  v2 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 96LL) + 8LL);
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
          for ( i = *(_QWORD *)(v10 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
            ;
          if ( (*(_BYTE *)(v10 + 193) & 0x10) == 0 && (unsigned int)sub_72F3C0(i, a1, 0, 0, 1) )
          {
            v5 = sub_8D7760(v10, a1, v12, v13, v14, v15);
            if ( !v5 )
              return v5;
            v5 = 1;
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
        for ( j = *(_BYTE *)(v7 + 140); j == 12; j = *(_BYTE *)(v7 + 140) )
          v7 = *(_QWORD *)(v7 + 160);
        if ( (unsigned __int8)(j - 9) <= 2u && !(unsigned int)sub_70D330(v7) )
          break;
      }
      v6 = *(_QWORD *)(v6 + 112);
      if ( !v6 )
        goto LABEL_24;
    }
  }
  else
  {
LABEL_24:
    v16 = **(__int64 ****)(a1 + 168);
    if ( !v16 )
      return 1;
    while ( 1 )
    {
      while ( 1 )
      {
        if ( ((_BYTE)v16[12] & 1) != 0 )
        {
          v17 = v16[5];
          for ( k = *((_BYTE *)v17 + 140); k == 12; k = *((_BYTE *)v17 + 140) )
            v17 = (__int64 *)v17[20];
          if ( (unsigned __int8)(k - 9) <= 2u )
            break;
        }
        v16 = (__int64 **)*v16;
        if ( !v16 )
          return 1;
      }
      if ( !(unsigned int)sub_70D330(v17) )
        break;
      v16 = (__int64 **)*v16;
      if ( !v16 )
        return 1;
    }
  }
  return 0;
}
