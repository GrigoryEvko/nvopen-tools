// Function: sub_2856C60
// Address: 0x2856c60
//
char __fastcall sub_2856C60(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  unsigned __int16 v8; // ax
  __int64 **v9; // rax
  int v10; // eax
  char v11; // dl
  __int16 v12; // ax
  __int64 v13; // rax
  _QWORD *v14; // r13
  _QWORD *v15; // r12
  _QWORD *v16; // rax
  __int64 v17; // rbx
  __int64 v18; // r13

  v6 = a3;
  while ( 1 )
  {
    v8 = *((_WORD *)a1 + 12);
    if ( v8 <= 4u )
    {
      while ( v8 > 1u )
      {
        a1 = (__int64 *)a1[4];
        v8 = *((_WORD *)a1 + 12);
        if ( v8 > 4u )
          goto LABEL_5;
      }
      goto LABEL_11;
    }
LABEL_5:
    if ( v8 == 15 )
      goto LABEL_11;
    if ( !*(_BYTE *)(a2 + 28) )
      goto LABEL_12;
    v9 = *(__int64 ***)(a2 + 8);
    a4 = *(unsigned int *)(a2 + 20);
    a3 = (__int64)&v9[a4];
    if ( v9 != (__int64 **)a3 )
    {
      while ( a1 != *v9 )
      {
        if ( (__int64 **)a3 == ++v9 )
          goto LABEL_27;
      }
      goto LABEL_11;
    }
LABEL_27:
    if ( (unsigned int)a4 < *(_DWORD *)(a2 + 16) )
    {
      a4 = (unsigned int)(a4 + 1);
      *(_DWORD *)(a2 + 20) = a4;
      *(_QWORD *)a3 = a1;
      ++*(_QWORD *)a2;
    }
    else
    {
LABEL_12:
      sub_C8CC70(a2, (__int64)a1, a3, a4, a5, a6);
      if ( !v11 )
        goto LABEL_11;
    }
    v12 = *((_WORD *)a1 + 12);
    if ( v12 == 5 )
    {
      v13 = a1[4];
      v14 = (_QWORD *)(v13 + 8 * a1[5]);
      if ( (_QWORD *)v13 != v14 )
      {
        v15 = (_QWORD *)a1[4];
        while ( !(unsigned __int8)sub_2856C60(*v15, a2, v6) )
        {
          if ( v14 == ++v15 )
            goto LABEL_11;
        }
LABEL_26:
        LOBYTE(v10) = 1;
        return v10;
      }
LABEL_11:
      LOBYTE(v10) = 0;
      return v10;
    }
    if ( v12 != 6 )
    {
LABEL_24:
      if ( v12 != 8 )
        goto LABEL_26;
      return (unsigned int)sub_2851300((__int64)a1, v6) ^ 1;
    }
    if ( a1[5] != 2 )
      goto LABEL_26;
    v16 = (_QWORD *)a1[4];
    a3 = v16[1];
    if ( *(_WORD *)(*v16 + 24LL) )
      break;
    a1 = (__int64 *)v16[1];
  }
  if ( *(_WORD *)(a3 + 24) != 15 )
    goto LABEL_26;
  v17 = *(_QWORD *)(*(_QWORD *)(a3 - 8) + 16LL);
  if ( !v17 )
    goto LABEL_26;
  while ( 1 )
  {
    v18 = *(_QWORD *)(v17 + 24);
    if ( *(_BYTE *)v18 == 46 && sub_D97040(v6, *(_QWORD *)(v18 + 8)) )
      break;
    v17 = *(_QWORD *)(v17 + 8);
    if ( !v17 )
    {
      v12 = *((_WORD *)a1 + 12);
      goto LABEL_24;
    }
  }
  LOBYTE(v10) = a1 == sub_DD8400(v6, v18);
  return v10;
}
