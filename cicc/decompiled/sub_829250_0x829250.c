// Function: sub_829250
// Address: 0x829250
//
_BOOL8 __fastcall sub_829250(__int64 a1, _BYTE *a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 v6; // r13
  _BOOL8 v7; // rsi
  unsigned int v8; // eax
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 **v13; // rcx
  __int64 *v14; // rbx
  char v15; // dl
  int v16; // r13d
  int v17; // r15d
  char v18; // al
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 **v22; // [rsp+8h] [rbp-38h]
  __int64 **v23; // [rsp+8h] [rbp-38h]

  v5 = a1;
  if ( !a3 )
    goto LABEL_8;
  if ( (*(_BYTE *)(a3 + 194) & 0x40) == 0 )
    goto LABEL_8;
  v6 = *(_QWORD *)(*(_QWORD *)(a3 + 40) + 32LL);
  if ( !*(_QWORD *)(*(_QWORD *)(*(_QWORD *)v6 + 96LL) + 56LL) )
    goto LABEL_8;
  v7 = 0;
  if ( (unsigned __int8)(*(_BYTE *)(v6 + 140) - 9) <= 2u )
    v7 = (*(_DWORD *)(v6 + 176) & 0x11000) == 4096;
  sub_866000(v6, v7, 1);
  v8 = sub_72F500(a3, v6, 0, 1, 1);
  v9 = v8;
  sub_5EAA30(a3, v8);
  sub_866010(a3, v9, v10, v11, v12);
  if ( *(_BYTE *)(a1 + 140) == 12 )
  {
    do
    {
      v5 = *(_QWORD *)(v5 + 160);
LABEL_8:
      ;
    }
    while ( *(_BYTE *)(v5 + 140) == 12 );
  }
  v13 = *(__int64 ***)(v5 + 168);
  v14 = *v13;
  if ( a2 )
  {
    v15 = a2[8];
    v16 = 0;
    v17 = 0;
    while ( 1 )
    {
      while ( v15 == 2 )
      {
        while ( 1 )
        {
          if ( !*(_QWORD *)a2 )
            BUG();
          v15 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
          v16 = 1;
          if ( v15 == 3 )
            break;
          a2 = *(_BYTE **)a2;
          if ( v15 != 2 )
            goto LABEL_15;
        }
        v23 = v13;
        v21 = sub_6BBB10(a2);
        v13 = v23;
        v15 = *(_BYTE *)(v21 + 8);
        a2 = (_BYTE *)v21;
      }
LABEL_15:
      if ( !v14 )
        break;
      if ( (*((_BYTE *)v14 + 33) & 1) != 0 )
      {
        v14 = (__int64 *)*v14;
        if ( !v14 )
          goto LABEL_29;
        v17 = 1;
      }
      else
      {
        if ( !*(_QWORD *)a2 )
          goto LABEL_22;
        v15 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
        if ( v15 == 3 )
        {
          v22 = v13;
          v20 = sub_6BBB10(a2);
          v14 = (__int64 *)*v14;
          a2 = (_BYTE *)v20;
          if ( !v20 )
          {
            while ( v14 )
            {
LABEL_24:
              if ( (*((_BYTE *)v14 + 33) & 1) == 0 )
              {
                v18 = *((_BYTE *)v14 + 32);
                if ( a3 && *(_BYTE *)(a3 + 174) == 7 && (v18 & 4) != 0 )
                {
                  if ( v16 )
                    return (unsigned int)sub_8291E0(a3) == 0;
                  return 0;
                }
                if ( (v18 & 0x10) == 0 && !v14[5] )
                  return 1;
                goto LABEL_29;
              }
LABEL_22:
              v14 = (__int64 *)*v14;
            }
            goto LABEL_29;
          }
          v15 = *(_BYTE *)(v20 + 8);
          v13 = v22;
        }
        else
        {
          v14 = (__int64 *)*v14;
          a2 = *(_BYTE **)a2;
        }
      }
    }
    if ( ((_BYTE)v13[2] & 1) == 0 && !v17 )
      return 1;
LABEL_29:
    if ( !v16 )
      return 0;
    if ( a3 )
      return (unsigned int)sub_8291E0(a3) == 0;
    return 1;
  }
  else
  {
    if ( v14 )
    {
      v16 = 0;
      goto LABEL_24;
    }
    return 0;
  }
}
