// Function: sub_734AF0
// Address: 0x734af0
//
__int64 __fastcall sub_734AF0(__int64 a1)
{
  _QWORD *v1; // r12
  _QWORD *i; // rdx
  __int64 v4; // rdi
  _QWORD *v5; // rax
  _QWORD *v6; // rsi
  __int64 v7; // rcx
  _QWORD *j; // rsi
  __int64 v9; // rdi
  _QWORD *v10; // rax
  _QWORD *v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // r14
  __int64 result; // rax
  __int64 ii; // rbx
  __int64 v16; // r13
  __int64 **v17; // rax
  __int64 *k; // r13
  char v19; // al
  char v20; // al
  __int64 m; // rbx
  __int64 n; // rbx
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 **v25; // rax

  v1 = *(_QWORD **)(a1 + 168);
  for ( i = (_QWORD *)v1[18]; i; v1[18] = i )
  {
LABEL_4:
    v4 = *(_QWORD *)(i[1] + 168LL);
    if ( (*(_BYTE *)(v4 + 111) & 4) == 0 )
    {
      v5 = *(_QWORD **)(v4 + 128);
      v6 = 0;
      while ( v5 )
      {
        v7 = *v5;
        if ( a1 == v5[1] )
        {
          if ( !v6 )
          {
            *(_QWORD *)(v4 + 128) = v7;
            i = (_QWORD *)v1[18];
            break;
          }
          *v6 = v7;
          i = *(_QWORD **)v1[18];
          v1[18] = i;
          if ( i )
            goto LABEL_4;
          goto LABEL_11;
        }
        v6 = v5;
        v5 = (_QWORD *)*v5;
      }
    }
    i = (_QWORD *)*i;
  }
LABEL_11:
  for ( j = (_QWORD *)v1[17]; j; v1[17] = j )
  {
    v9 = j[1];
    if ( (*(_BYTE *)(v9 + 194) & 0x40) == 0 )
    {
      v10 = *(_QWORD **)(v9 + 232);
      v11 = 0;
      while ( v10 )
      {
        v12 = *v10;
        if ( a1 == v10[1] )
        {
          if ( v11 )
            *v11 = v12;
          else
            *(_QWORD *)(v9 + 232) = v12;
          j = (_QWORD *)v1[17];
          break;
        }
        v11 = v10;
        v10 = (_QWORD *)*v10;
      }
    }
    j = (_QWORD *)*j;
  }
  v13 = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 152LL);
  if ( !v13 || (*(_BYTE *)(v13 + 29) & 0x20) != 0 )
    goto LABEL_23;
  v16 = *(_QWORD *)(v13 + 144);
  if ( v16 )
  {
    do
    {
      sub_734AA0(v16);
      v16 = *(_QWORD *)(v16 + 112);
    }
    while ( v16 );
    v17 = *(__int64 ***)(a1 + 96);
    if ( v17 )
    {
      for ( k = *v17; k; k = (__int64 *)*k )
      {
LABEL_38:
        v19 = *((_BYTE *)k + 16);
        if ( v19 == 54 )
        {
          if ( a1 == *(_QWORD *)(k[3] + 16) )
            break;
        }
        else if ( v19 == 53 )
        {
          v23 = k[3];
          if ( *(_BYTE *)(v23 + 16) == 11 )
          {
            v24 = *(_QWORD *)(v23 + 32);
            if ( v24 )
              sub_734A70(v24);
          }
        }
      }
    }
    v13 = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 152LL);
    if ( !v13 )
      goto LABEL_23;
    goto LABEL_41;
  }
  v25 = *(__int64 ***)(a1 + 96);
  if ( v25 )
  {
    k = *v25;
    if ( *v25 )
      goto LABEL_38;
LABEL_41:
    if ( (*(_BYTE *)(v13 + 29) & 0x20) != 0 )
      goto LABEL_23;
  }
  v20 = *(_BYTE *)(a1 + 177);
  if ( (v20 & 0x10) != 0 )
  {
    if ( v20 < 0 )
      goto LABEL_23;
  }
  else if ( !*(_QWORD *)(v13 + 272) || v20 < 0 )
  {
    goto LABEL_23;
  }
  for ( m = *(_QWORD *)(v13 + 112); m; m = *(_QWORD *)(m + 112) )
  {
    while ( (*(_BYTE *)(m + 170) & 0xB0) != 0x10 || !*(_QWORD *)m )
    {
      m = *(_QWORD *)(m + 112);
      if ( !m )
        goto LABEL_50;
    }
    sub_8AD0D0(*(_QWORD *)m, 0, 2);
  }
LABEL_50:
  for ( n = *(_QWORD *)(v13 + 144); n; n = *(_QWORD *)(n + 112) )
  {
    if ( ((*(_BYTE *)(n + 195) & 3) == 1 || dword_4F068EC && *(char *)(n + 192) < 0)
      && (*(_BYTE *)(n + 195) & 8) == 0
      && *(_QWORD *)n )
    {
      sub_8AD0D0(*(_QWORD *)n, 0, 2);
    }
  }
LABEL_23:
  result = v1[19];
  if ( result )
  {
    if ( (*(_BYTE *)(result + 29) & 0x20) == 0 )
    {
      for ( ii = *(_QWORD *)(result + 104); ii; ii = *(_QWORD *)(ii + 112) )
      {
        result = (unsigned int)*(unsigned __int8 *)(ii + 140) - 9;
        if ( (unsigned __int8)(*(_BYTE *)(ii + 140) - 9) <= 2u )
          result = sub_734AF0(ii);
      }
    }
  }
  return result;
}
