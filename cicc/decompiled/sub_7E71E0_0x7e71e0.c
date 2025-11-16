// Function: sub_7E71E0
// Address: 0x7e71e0
//
__int64 __fastcall sub_7E71E0(__int64 a1, __int64 a2, int a3)
{
  unsigned int v3; // r10d
  __int64 v4; // r14
  __int64 v5; // rcx
  __int64 i; // rdx
  int v7; // r9d
  __int64 v8; // r12
  bool v9; // cl
  __int64 j; // r15
  bool v11; // bl
  __int64 v13; // r14
  _QWORD **v14; // rax
  __int64 v15; // r14
  __int64 v16; // rdi
  _QWORD **v17; // rax
  unsigned int v18; // [rsp+8h] [rbp-48h]
  int v19; // [rsp+Ch] [rbp-44h]
  int v21; // [rsp+18h] [rbp-38h]
  unsigned int v22; // [rsp+18h] [rbp-38h]

  v3 = 0;
  if ( !a1 )
    return v3;
  v4 = qword_4D03F68[5];
  if ( v4 && (*(_BYTE *)(v4 + 49) & 8) != 0 )
  {
    v5 = 0;
    for ( i = *(_QWORD *)(*(_QWORD *)(v4 + 24) + 24LL); ; i = *(_QWORD *)(i + 32) )
    {
      if ( (*(_BYTE *)(i + 49) & 8) != 0 )
      {
        if ( !v5 )
          v5 = i;
        if ( v4 == i )
        {
LABEL_11:
          v4 = v5;
          break;
        }
      }
      else
      {
        v5 = 0;
        if ( v4 == i )
          goto LABEL_11;
      }
    }
  }
  v7 = 0;
  v8 = 0;
  v9 = 0;
  v21 = 0;
  v3 = 0;
  for ( j = qword_4F06BC0; ; j = *(_QWORD *)(j + 32) )
  {
    if ( v4 )
    {
      v11 = v9;
      while ( 1 )
      {
        if ( (*(_BYTE *)(v4 + 49) & 0x10) != 0 && v11
          || (*(_BYTE *)(v4 + 49) & 0x20) != 0
          || (*(_BYTE *)(v4 + 50) & 1) != 0 )
        {
          goto LABEL_16;
        }
        if ( (*(_DWORD *)(v4 + 48) & 0x20024000) != 0 )
          break;
        if ( *(_QWORD *)(v4 + 8) || !unk_4D03F58 )
        {
          if ( a3 )
            return 1;
LABEL_24:
          if ( v7 )
          {
            qword_4D03F68[6] = v8;
            nullsub_11();
          }
          sub_7FE8B0(v4, a2);
          v4 = *(_QWORD *)(v4 + 32);
          v7 = 0;
          v3 = 1;
          if ( !v4 )
            goto LABEL_27;
        }
        else
        {
          if ( unk_4D03F58 != *(_QWORD *)(*(_QWORD *)(v4 + 80) + 16LL) )
          {
            if ( a3 )
              return 1;
            goto LABEL_24;
          }
LABEL_16:
          v4 = *(_QWORD *)(v4 + 32);
          if ( !v4 )
            goto LABEL_27;
        }
      }
      v7 = 1;
      v8 = *(_QWORD *)(*(_QWORD *)(v4 + 80) + 80LL);
      goto LABEL_16;
    }
LABEL_27:
    if ( *(_BYTE *)j != 5 )
    {
      v21 = 0;
      if ( *(_BYTE *)(j + 8) == 23 )
      {
        v15 = *(_QWORD *)(j + 16);
        if ( *(_BYTE *)(v15 + 28) == 2 )
        {
          v16 = *(_QWORD *)(v15 + 32);
          if ( v16 )
          {
            if ( a3 )
              return 1;
            if ( v7 )
            {
              qword_4D03F68[6] = v8;
              nullsub_11();
              v16 = *(_QWORD *)(v15 + 32);
            }
            sub_7DE0A0(v16, a2);
            v21 = 1;
            v7 = 0;
            v3 = 1;
          }
        }
      }
      goto LABEL_29;
    }
    if ( *(_BYTE *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 174LL) == 2 )
    {
      v19 = v7;
      v18 = v3;
      v17 = sub_7E71B0((_QWORD *)j);
      v7 = v19;
      if ( *((_BYTE *)v17 + 25) )
      {
        if ( !v21 )
        {
          v21 = 0;
          v3 = v18;
LABEL_29:
          if ( j == a1 )
            break;
          goto LABEL_30;
        }
      }
    }
    if ( a3 )
      return 1;
    if ( v7 )
    {
      qword_4D03F68[6] = v8;
      nullsub_11();
    }
    v13 = *(_QWORD *)(j + 16);
    v14 = sub_7E71B0((_QWORD *)j);
    sub_7DE060((__int64)v14, v13, a2);
    v21 = 0;
    v7 = 0;
    v3 = 1;
    if ( j == a1 )
      break;
LABEL_30:
    v4 = *(_QWORD *)(j + 40);
    v9 = *(_BYTE *)j == 2;
  }
  if ( v7 && (*(_BYTE *)j != 1 || *(_QWORD *)(j + 16) != qword_4F04C50) )
  {
    v22 = v7;
    if ( a3 )
    {
      return 1;
    }
    else
    {
      qword_4D03F68[6] = v8;
      nullsub_11();
      return v22;
    }
  }
  return v3;
}
