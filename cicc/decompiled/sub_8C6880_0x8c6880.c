// Function: sub_8C6880
// Address: 0x8c6880
//
_QWORD *__fastcall sub_8C6880(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  char v5; // al
  __int64 v6; // r13
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  _QWORD *v9; // rbx
  _QWORD *v10; // r14
  _QWORD *i; // rax
  _QWORD *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 *v15; // r9
  __int64 *v16; // r10
  char v17; // al
  __int64 v18; // r10
  __int64 v19; // rcx
  _UNKNOWN *__ptr32 *v20; // r8
  _QWORD *v21; // rax
  _QWORD *v22; // r13
  _QWORD *v23; // rax
  _QWORD *v24; // rdx
  __int64 v26; // [rsp+0h] [rbp-60h]
  __int64 v27; // [rsp+0h] [rbp-60h]
  __int64 v28; // [rsp+8h] [rbp-58h]
  __int64 v29; // [rsp+8h] [rbp-58h]
  _QWORD *v30; // [rsp+10h] [rbp-50h]
  __int64 v31; // [rsp+18h] [rbp-48h]
  __int64 v32; // [rsp+18h] [rbp-48h]
  _QWORD v33[8]; // [rsp+20h] [rbp-40h] BYREF

  v4 = a1;
  v5 = *(_BYTE *)(a2 + 80);
  v6 = *(_QWORD *)(a2 + 88);
  if ( v5 == 3 )
    goto LABEL_35;
  if ( v5 == 6 )
  {
    if ( dword_4F077C4 != 2 )
      goto LABEL_4;
    goto LABEL_35;
  }
  if ( dword_4F077C4 == 2 && (unsigned __int8)(v5 - 4) <= 2u )
LABEL_35:
    v4 = *(_QWORD *)(sub_892920(**(_QWORD **)(a1 + 104)) + 88);
LABEL_4:
  v30 = (_QWORD *)(v4 + 112);
  v7 = *(_QWORD **)(v4 + 112);
  if ( v7 )
  {
    do
    {
      v8 = v7;
      v7 = (_QWORD *)*v7;
    }
    while ( v7 );
    v9 = v8;
  }
  else
  {
    v9 = (_QWORD *)(v4 + 112);
  }
  v33[1] = a2;
  v33[0] = 0;
  *v9 = v33;
  v10 = *(_QWORD **)(v4 + 112);
  if ( v10 == v33 )
  {
LABEL_43:
    v10 = 0;
  }
  else
  {
    for ( i = v33; ; i = (_QWORD *)*v9 )
    {
      if ( i )
      {
        do
        {
          v12 = i;
          i = (_QWORD *)*i;
        }
        while ( i );
        v9 = v12;
      }
      *v9 = v10;
      *(_QWORD *)(v4 + 112) = *v10;
      v13 = v10[1];
      *v10 = 0;
      v14 = *(_QWORD *)(v13 + 88);
      if ( v6 == v14 )
        break;
      v15 = *(__int64 **)(v6 + 168);
      v16 = *(__int64 **)(v14 + 168);
      if ( (*(_BYTE *)(v4 + 265) & 1) != 0 )
      {
        if ( sub_89AB40(*v15, *v16, 32 * ((*(_BYTE *)(v4 + 160) & 8) != 0), a4, (_UNKNOWN *__ptr32 *)v14) )
          goto LABEL_24;
      }
      else
      {
        v17 = *(_BYTE *)(v14 + 177) ^ *(_BYTE *)(v6 + 177);
        v31 = v14;
        if ( (v17 & 0x20) == 0 && v17 >= 0 )
        {
          v26 = *(_QWORD *)(v14 + 168);
          v28 = *(_QWORD *)(v6 + 168);
          if ( sub_89AB40(v15[21], v16[21], 2, a4, (_UNKNOWN *__ptr32 *)v14) )
          {
            if ( (v18 = v26, !*(_QWORD *)(v28 + 176)) && !*(_QWORD *)(v26 + 176)
              || (v27 = v28, v29 = v31, v32 = v18, (unsigned int)sub_8D2490(v6))
              || (unsigned int)sub_8D2490(v29)
              || sub_89AB40(*(_QWORD *)(v27 + 176), *(_QWORD *)(v32 + 176), 2, v19, v20) )
            {
LABEL_24:
              v21 = *(_QWORD **)(v4 + 112);
              goto LABEL_25;
            }
          }
        }
      }
      v10 = *(_QWORD **)(v4 + 112);
      if ( v10 == v33 )
        goto LABEL_43;
    }
    v21 = *(_QWORD **)(v4 + 112);
    v10 = 0;
LABEL_25:
    if ( v21 != v33 )
    {
      do
      {
        v22 = v21;
        v21 = (_QWORD *)*v21;
      }
      while ( v21 != v33 );
      v30 = v22;
    }
  }
  if ( v33[0] )
  {
    v23 = (_QWORD *)*v9;
    if ( *v9 )
    {
      do
      {
        v24 = v23;
        v23 = (_QWORD *)*v23;
      }
      while ( v23 );
      v9 = v24;
    }
    *v30 = 0;
    *v9 = *(_QWORD *)(v4 + 112);
    *(_QWORD *)(v4 + 112) = v33[0];
  }
  else
  {
    *v30 = 0;
  }
  return v10;
}
