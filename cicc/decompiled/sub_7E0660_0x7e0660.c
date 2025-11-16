// Function: sub_7E0660
// Address: 0x7e0660
//
_QWORD *__fastcall sub_7E0660(__int64 a1, __int64 a2, _QWORD *a3, char a4)
{
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 i; // r15
  _QWORD *v9; // r14
  __int64 v10; // r13
  _QWORD *v11; // rdx
  __int64 j; // r14
  __int64 v13; // rsi
  _QWORD *v15; // rax
  bool v16; // dl
  __int64 v17; // rdx
  __int64 v18; // [rsp+10h] [rbp-50h]
  __int64 v19; // [rsp+20h] [rbp-40h]
  __int64 v20; // [rsp+28h] [rbp-38h]

  v19 = *(_QWORD *)(a1 + 168);
  if ( a2 )
  {
    v18 = *(_QWORD *)(a2 + 40);
    v20 = *(_QWORD *)(v18 + 168);
  }
  else
  {
    v20 = *(_QWORD *)(a1 + 168);
    v18 = a1;
  }
  v6 = *(_QWORD *)(v20 + 24);
  if ( v6 && ((*(_BYTE *)(v6 + 96) & 2) == 0 || (a4 & 1) != 0) )
  {
    if ( a2 )
      v6 = sub_8E5650(*(_QWORD *)(v20 + 24));
    a3 = (_QWORD *)sub_7E0660(a1, v6);
  }
  v7 = *(_QWORD *)(v20 + 152);
  if ( v7 )
  {
    for ( i = *(_QWORD *)(v7 + 144); i; i = *(_QWORD *)(i + 112) )
    {
      while ( (*(_DWORD *)(i + 192) & 0x8000400) != 0 || (*(_BYTE *)(i + 192) & 2) == 0 )
      {
LABEL_12:
        i = *(_QWORD *)(i + 112);
        if ( !i )
          goto LABEL_26;
      }
      v9 = *(_QWORD **)(v19 + 64);
      if ( !v9 )
      {
LABEL_34:
        v15 = sub_725010();
        v16 = 0;
        v15[1] = i;
        v15[2] = a2;
        if ( a2 )
        {
          v16 = 1;
          if ( (*(_BYTE *)(a2 + 96) & 2) == 0 )
            v16 = (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL) + 16LL) + 96LL) & 2) != 0;
        }
        *((_BYTE *)v15 + 32) = v16;
        v17 = *(_QWORD *)(v19 + 48);
        *(_QWORD *)(v19 + 48) = v17 - 1;
        v15[3] = v17;
        *a3 = v15;
        a3 = v15;
        goto LABEL_12;
      }
      while ( 1 )
      {
        v10 = v9[1];
        v11 = *(_QWORD **)v10;
        if ( *(_QWORD *)i )
        {
          if ( v11
            && (**(_QWORD **)i == *v11 || *(_BYTE *)(i + 174) == 2 && *(_BYTE *)(v10 + 174) == 2)
            && (unsigned int)sub_8DE890(*(_QWORD *)(i + 152), *(_QWORD *)(v10 + 152), 0, 0)
            && (unsigned int)sub_8D7820(*(_QWORD *)(i + 152), *(_QWORD *)(v10 + 152), 0, 0) )
          {
            break;
          }
        }
        v9 = (_QWORD *)*v9;
        if ( !v9 )
          goto LABEL_34;
      }
    }
  }
LABEL_26:
  for ( j = *(_QWORD *)(*(_QWORD *)(v18 + 168) + 8LL); j; j = *(_QWORD *)(j + 8) )
  {
    if ( (*(_BYTE *)(j + 96) & 2) == 0 && *(_QWORD *)(v20 + 24) != j )
    {
      v13 = j;
      if ( a2 )
        v13 = sub_8E5650(j);
      a3 = (_QWORD *)sub_7E0660(a1, v13);
    }
  }
  return a3;
}
