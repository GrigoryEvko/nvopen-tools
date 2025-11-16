// Function: sub_8769C0
// Address: 0x8769c0
//
__int16 __fastcall sub_8769C0(
        __int64 a1,
        FILE *a2,
        __int64 a3,
        int a4,
        __int64 a5,
        __int64 a6,
        unsigned int a7,
        unsigned int a8,
        _DWORD *a9)
{
  __int64 v9; // r12
  char v10; // al
  __int64 v11; // rcx
  __int64 v12; // r13
  __int64 v13; // rax
  int v14; // r9d
  char v15; // al
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v23; // [rsp-10h] [rbp-60h]
  unsigned int v25; // [rsp+14h] [rbp-3Ch]
  int v27; // [rsp+1Ch] [rbp-34h]

  v9 = a1;
  v10 = *(_BYTE *)(a1 + 80);
  v11 = a7;
  v27 = a5;
  v25 = a6;
  if ( v10 == 16 )
  {
    v9 = **(_QWORD **)(a1 + 88);
    v10 = *(_BYTE *)(v9 + 80);
  }
  if ( v10 == 24 )
    v9 = *(_QWORD *)(v9 + 88);
  v12 = *(_QWORD *)(v9 + 88);
  if ( !a7 )
  {
LABEL_6:
    if ( a9 )
      goto LABEL_7;
    goto LABEL_23;
  }
  if ( (unsigned int)sub_884000(a1, 1) )
  {
    if ( !a3 )
      goto LABEL_6;
    v15 = *(_BYTE *)(a1 + 80);
    if ( v15 == 16 )
    {
      a1 = **(_QWORD **)(a1 + 88);
      v15 = *(_BYTE *)(a1 + 80);
    }
    if ( v15 == 24 )
    {
      a1 = *(_QWORD *)(a1 + 88);
      v15 = *(_BYTE *)(a1 + 80);
    }
    if ( (unsigned __int8)(v15 - 10) <= 1u )
    {
      v19 = *(_QWORD *)(a1 + 88);
      if ( (*(_BYTE *)(v19 + 194) & 0x40) == 0 )
        goto LABEL_33;
      do
        v19 = *(_QWORD *)(v19 + 232);
      while ( (*(_BYTE *)(v19 + 194) & 0x40) != 0 );
    }
    else
    {
      if ( v15 != 20 )
        goto LABEL_33;
      v18 = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 176LL);
      if ( (*(_BYTE *)(v18 + 194) & 0x40) == 0 )
        goto LABEL_33;
      do
        v18 = *(_QWORD *)(v18 + 232);
      while ( (*(_BYTE *)(v18 + 194) & 0x40) != 0 );
      v19 = *(_QWORD *)(v18 + 248);
    }
    a1 = *(_QWORD *)v19;
LABEL_33:
    sub_8843A0(a1, a1, a2, a3, a9);
    goto LABEL_6;
  }
  v14 = 7;
  if ( qword_4D0495C )
  {
    if ( *(_BYTE *)(v12 + 174) == 2 )
    {
      v20 = *(_QWORD *)(a1 + 64);
      if ( a3 != v20 )
      {
        if ( a3 && v20 )
        {
          v14 = 5;
          if ( dword_4F07588 )
          {
            v21 = *(_QWORD *)(v20 + 32);
            if ( *(_QWORD *)(a3 + 32) != v21 || (v14 = 7, !v21) )
              v14 = 5;
          }
        }
        else
        {
          v14 = a3 == 0 ? 7 : 5;
        }
      }
    }
  }
  sub_87D9B0(a1, 0, 0, (_DWORD)a2, 0, v14, 330, (__int64)a9);
  v11 = v23;
  if ( a9 )
  {
LABEL_7:
    if ( (*(_BYTE *)(v12 + 206) & 0x10) != 0 )
      *a9 = 1;
    v13 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( *(char *)(v13 + 11) >= 0 || *(_BYTE *)(v12 + 174) != 1 )
      return v13;
    goto LABEL_12;
  }
LABEL_23:
  sub_8767A0(132, v9, a2, 0);
  v13 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( *(char *)(v13 + 11) >= 0 || *(_BYTE *)(v12 + 174) != 1 )
  {
LABEL_14:
    if ( v27 )
    {
      if ( !a9 )
      {
        if ( (*(_BYTE *)(v12 + 192) & 2) == 0 || (LOWORD(v13) = a4, !a4) )
          LOWORD(v13) = sub_732910(v12, v25, a8, v11, a5, a6);
      }
    }
    return v13;
  }
LABEL_12:
  if ( (*(_BYTE *)(v12 + 193) & 0x10) == 0 && (*(_BYTE *)(v12 + 206) & 8) == 0 )
    goto LABEL_14;
  LODWORD(v13) = sub_72F310(v12, 0);
  if ( !(_DWORD)v13 )
    goto LABEL_14;
  v11 = qword_4F04C68[0];
  a5 = *(_QWORD *)(*(_QWORD *)(v12 + 40) + 32LL);
  v13 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  while ( 1 )
  {
    if ( (unsigned __int8)(*(_BYTE *)(v13 + 4) - 6) <= 1u )
    {
      v16 = *(_QWORD *)(v13 + 208);
      if ( v16 == a5 )
        break;
      if ( a5 )
      {
        if ( v16 )
        {
          if ( dword_4F07588 )
          {
            v17 = *(_QWORD *)(v16 + 32);
            if ( *(_QWORD *)(a5 + 32) == v17 )
            {
              if ( v17 )
                break;
            }
          }
        }
      }
    }
    v13 = qword_4F04C68[0] + 776LL * *(int *)(v13 + 552);
    if ( *(char *)(v13 + 11) >= 0 )
      goto LABEL_14;
  }
  if ( a9 )
    *a9 = 1;
  else
    LOWORD(v13) = sub_685360(0x974u, a2, *(_QWORD *)(*(_QWORD *)(v12 + 40) + 32LL));
  return v13;
}
