// Function: sub_866C00
// Address: 0x866c00
//
__int64 __fastcall sub_866C00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 i; // rdx
  __int64 *v10; // r12
  __int64 **v11; // rbx
  int v12; // r14d
  __int64 v13; // rax
  __int64 v14; // rax
  char v15; // dl
  __int64 *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax

  result = 0;
  if ( !a1 )
    return result;
  v7 = a1;
  if ( *(_BYTE *)(a1 + 42) )
    return result;
  v8 = *(_QWORD *)(a1 + 16);
  if ( !v8 )
    goto LABEL_20;
  i = *(_QWORD *)(a1 + 8);
  v10 = *(__int64 **)(v8 + 8);
  v11 = *(__int64 ***)(i + 24);
  if ( !v11 )
    goto LABEL_69;
  v12 = 0;
  do
  {
    while ( 1 )
    {
      while ( 1 )
      {
        i = *((unsigned int *)v11 + 8);
        v13 = v10[10];
        if ( (_DWORD)i != 1 )
          break;
        if ( v13 )
        {
          if ( (*(_BYTE *)(v13 + 175) & 0x20) != 0 )
          {
            i = *(_QWORD *)(v7 + 16);
            if ( *(_BYTE *)(i + 18) )
              goto LABEL_16;
          }
          i = *(_QWORD *)(v13 + 112);
          if ( i )
          {
            a4 = *(_QWORD *)(i + 128);
            if ( a4 )
            {
              a2 = *(unsigned int *)(a4 + 36);
              if ( *(_DWORD *)(*(_QWORD *)(v13 + 128) + 36LL) == (_DWORD)a2 )
              {
                v17 = v10[5];
                v10[10] = i;
                *(_QWORD *)(v17 + 88) = i;
                goto LABEL_16;
              }
            }
          }
        }
        v10[10] = 0;
        v11 = (__int64 **)*v11;
        v12 = 1;
        v10 = (__int64 *)*v10;
        if ( !v11 )
          goto LABEL_17;
      }
      if ( !(_DWORD)i )
        break;
      if ( (_DWORD)i == 3 )
      {
        if ( !v13 )
          goto LABEL_60;
        if ( (*(_BYTE *)(v13 + 145) & 2) == 0 || (i = *(_QWORD *)(v7 + 16), !*(_BYTE *)(i + 18)) )
        {
          i = *(_QWORD *)(v13 + 112);
          a4 = v10[5];
          if ( !i || (a2 = *(_QWORD *)(i + 8), *(_QWORD *)(v13 + 8) != a2) )
          {
            v10[10] = 0;
            v16 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(v13 + 40) + 32LL) + 160LL);
            for ( i = *v16; a4 != *v16; i = *v16 )
              v16 = (__int64 *)v16[14];
            *(_QWORD *)(i + 88) = v16;
            goto LABEL_60;
          }
          v10[10] = i;
          *(_QWORD *)(a4 + 88) = i;
        }
LABEL_16:
        v11 = (__int64 **)*v11;
        v10 = (__int64 *)*v10;
        if ( !v11 )
          goto LABEL_17;
      }
      else
      {
        if ( (_DWORD)i == 4 )
        {
          if ( (*(_BYTE *)(v13 + 24) & 0x10) == 0 || (i = *(_QWORD *)(v7 + 16), !*(_BYTE *)(i + 18)) )
          {
            v18 = *(_QWORD *)v13;
            v10[10] = v18;
            if ( !v18 )
              goto LABEL_60;
          }
          goto LABEL_16;
        }
        if ( !*(_BYTE *)(v7 + 40) )
        {
          i = *(_QWORD *)v13;
          v10[10] = *(_QWORD *)v13;
          if ( (*(_BYTE *)(v13 + 42) & 1) == 0 || (a4 = *(_QWORD *)(v7 + 16), !*(_BYTE *)(a4 + 18)) )
          {
            if ( !i )
              goto LABEL_60;
            a4 = *(unsigned int *)(i + 120);
            if ( *(_DWORD *)(v13 + 120) != (_DWORD)a4 )
              goto LABEL_60;
            *(_QWORD *)(v10[5] + 88) = i;
          }
          goto LABEL_16;
        }
        if ( v13 )
        {
          i = *(_QWORD *)v13;
          if ( (*(_BYTE *)(v13 + 33) & 1) != 0 )
          {
            a4 = *(_QWORD *)(v7 + 16);
            if ( *(_BYTE *)(a4 + 18) )
              goto LABEL_39;
          }
          if ( !i )
            goto LABEL_38;
          if ( (*(_BYTE *)(i + 33) & 2) != 0 )
          {
            a4 = *(unsigned int *)(i + 36);
            if ( *(_DWORD *)(v13 + 36) == (_DWORD)a4 )
              goto LABEL_39;
          }
        }
        i = 0;
LABEL_38:
        v12 = 1;
LABEL_39:
        v10[10] = i;
        v11 = (__int64 **)*v11;
        v10 = (__int64 *)*v10;
        if ( !v11 )
          goto LABEL_17;
      }
    }
    if ( !v13 )
      goto LABEL_41;
    i = *(unsigned __int8 *)(v13 + 24);
    a1 = (__int64)v11[1];
    if ( (i & 0x10) != 0 && (a4 = *(_QWORD *)(v7 + 16), *(_BYTE *)(a4 + 18)) )
    {
      a2 = v10[10];
    }
    else
    {
      a2 = *(_QWORD *)v13;
      v10[10] = *(_QWORD *)v13;
      if ( !a2 )
      {
LABEL_41:
        a2 = 0;
        goto LABEL_42;
      }
      LOBYTE(i) = *(_BYTE *)(a2 + 24);
    }
    i &= 8u;
    if ( (_DWORD)i )
    {
      if ( !*(_WORD *)(v7 + 40) )
        sub_85BC00(a1, a2);
      goto LABEL_16;
    }
LABEL_42:
    if ( !*(_BYTE *)(v7 + 40) )
      goto LABEL_60;
    i = *(unsigned int *)(v7 + 48);
    if ( !(_DWORD)i || *((_BYTE *)v10 + 96) || a2 )
      goto LABEL_60;
    i = *(_QWORD *)(v7 + 16);
    if ( *(__int64 **)(i + 8) != v10 || *v10 )
    {
      if ( v13 )
        goto LABEL_48;
      goto LABEL_16;
    }
    if ( v13 )
    {
      v12 = 1;
LABEL_48:
      a4 = v10[9];
      v14 = *(_QWORD *)(a4 + 8);
      v15 = *(_BYTE *)(v14 + 80);
      if ( v15 == 3 || v15 == 2 )
      {
        i = *(_QWORD *)(a4 + 64);
        *(_QWORD *)(v14 + 88) = i;
      }
      else
      {
        i = *(_QWORD *)(v14 + 88);
        *(_QWORD *)(i + 200) = v14;
        *(_QWORD *)(i + 208) = 0;
      }
      *(_BYTE *)(v14 + 82) &= ~1u;
      goto LABEL_16;
    }
LABEL_60:
    v11 = (__int64 **)*v11;
    v10 = (__int64 *)*v10;
    v12 = 1;
  }
  while ( v11 );
LABEL_17:
  if ( !v12 )
  {
LABEL_69:
    if ( !*(_BYTE *)(v7 + 40) )
      sub_7BC300(*(_QWORD *)(v7 + 24), (unsigned int *)a2, i, a4, a5, a6);
    *(_BYTE *)(*(_QWORD *)(v7 + 16) + 16LL) = 1;
    return 1;
  }
  if ( !*(_BYTE *)(v7 + 40) )
    sub_7AEC00(a1, a2);
LABEL_20:
  sub_85FCF0();
  return 0;
}
