// Function: sub_5EBC80
// Address: 0x5ebc80
//
_QWORD *__fastcall sub_5EBC80(__int64 a1, __int64 a2, _QWORD *a3, _QWORD *a4, __int64 *a5, __int64 a6)
{
  char v10; // r8
  __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  char v16; // r8
  __int64 v17; // rbx
  __int64 v18; // rax
  char v19; // al
  char v20; // dl
  _QWORD *v21; // rax
  __int64 v22; // rcx
  _QWORD *v23; // r14
  _QWORD *v24; // rdx
  _QWORD *v25; // r12
  __int64 *v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 i; // r15
  __int64 *v30; // rax
  _QWORD *result; // rax
  __int64 j; // r13
  __int64 *v33; // r15
  __int64 v34; // rdi
  char v35; // [rsp+14h] [rbp-3Ch]
  unsigned int v36; // [rsp+14h] [rbp-3Ch]

  v10 = *(_BYTE *)(*(_QWORD *)(a1 + 112) + 25LL);
  if ( (*(_BYTE *)(a1 + 96) & 2) != 0 && (v11 = **(_QWORD **)(a6 + 168)) != 0 )
  {
    while ( 1 )
    {
      if ( (*(_BYTE *)(v11 + 96) & 2) != 0 )
      {
        v12 = *(_QWORD *)(v11 + 40);
        v13 = *(_QWORD *)(a1 + 40);
        if ( v12 == v13 )
          break;
        if ( v13 )
        {
          if ( v12 )
          {
            if ( dword_4F07588 )
            {
              v14 = *(_QWORD *)(v12 + 32);
              if ( *(_QWORD *)(v13 + 32) == v14 )
              {
                if ( v14 )
                  break;
              }
            }
          }
        }
      }
      v11 = *(_QWORD *)v11;
      if ( !v11 )
        goto LABEL_11;
    }
    if ( !v10 && (*(_BYTE *)(a2 + 97) & 1) != 0 )
      *(_BYTE *)(v11 + 97) |= 1u;
    return sub_5EBB30(v11, a3, a4, v10);
  }
  else
  {
LABEL_11:
    v35 = v10;
    v15 = sub_725160();
    v16 = v35;
    v17 = v15;
    v18 = *(_QWORD *)(a1 + 40);
    *(_QWORD *)(v17 + 56) = a6;
    *(_QWORD *)(v17 + 40) = v18;
    *(_QWORD *)(v17 + 72) = *(_QWORD *)(a2 + 72);
    v19 = *(_BYTE *)(v17 + 96) & 0xFE;
    *(_BYTE *)(v17 + 96) = v19;
    if ( (*(_BYTE *)(a1 + 96) & 2) != 0 )
      *(_BYTE *)(v17 + 96) = v19 | 2;
    v20 = 0;
    if ( !v35 )
      v20 = *(_BYTE *)(a2 + 97) & 1;
    *(_BYTE *)(v17 + 97) = v20 | *(_BYTE *)(v17 + 97) & 0xFE;
    if ( (*(_BYTE *)(a1 + 96) & 2) == 0 )
    {
      sub_8E52B0(a1, a6, a2, v17);
      v16 = v35;
    }
    v21 = sub_5EBB30(v17, a3, a4, v16);
    v22 = *(_QWORD *)(v17 + 40);
    v23 = v21;
    v25 = v24;
    v26 = **(__int64 ***)(a6 + 168);
    if ( v26 )
    {
      while ( 1 )
      {
        v28 = v26[5];
        if ( v28 == v22 )
          break;
        if ( v28 )
        {
          if ( v22 )
          {
            if ( dword_4F07588 )
            {
              v27 = *(_QWORD *)(v28 + 32);
              if ( *(_QWORD *)(v22 + 32) == v27 )
              {
                if ( v27 )
                  break;
              }
            }
          }
        }
        v26 = (__int64 *)*v26;
        if ( !v26 )
          goto LABEL_27;
      }
      *((_BYTE *)v26 + 96) |= 4u;
      v22 = *(_QWORD *)(v17 + 40);
      *(_BYTE *)(v17 + 96) |= 4u;
    }
LABEL_27:
    v36 = 0;
    for ( i = *(_QWORD *)(*(_QWORD *)(v22 + 168) + 8LL); i; i = *(_QWORD *)(i + 8) )
    {
      if ( (*(_BYTE *)(i + 96) & 2) != 0 && (*(_BYTE *)(*(_QWORD *)(i + 112) + 24LL) & 1) == 0 )
        v36 = 1;
      else
        sub_5EBC80(i, v17, v23, v25, a5, a6);
    }
    v30 = (__int64 *)*a5;
    if ( !*a5 )
      v30 = *(__int64 **)(a6 + 168);
    *v30 = v17;
    *a5 = v17;
    sub_5E6390(a6, v17);
    result = (_QWORD *)v36;
    if ( v36 )
    {
      result = *(_QWORD **)(*(_QWORD *)(v17 + 40) + 168LL);
      for ( j = result[1]; j; j = *(_QWORD *)(j + 8) )
      {
        if ( (*(_BYTE *)(j + 96) & 2) != 0 )
        {
          v33 = *(__int64 **)(j + 112);
          if ( (v33[3] & 1) == 0 )
          {
            v34 = sub_8E5650(j);
            do
              v33 = (__int64 *)*v33;
            while ( (v33[3] & 1) == 0 );
            result = sub_5EBB30(v34, v23, v25, *((_BYTE *)v33 + 25));
          }
        }
      }
    }
  }
  return result;
}
