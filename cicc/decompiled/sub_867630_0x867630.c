// Function: sub_867630
// Address: 0x867630
//
__int64 __fastcall sub_867630(unsigned __int64 a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  unsigned __int64 v7; // rbx
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rax
  char v15; // r9
  char v16; // r9
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 *v19; // rax
  __int64 v20; // rdx
  unsigned int v21; // r10d
  unsigned int *v22; // rax
  unsigned int *v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax

  v6 = 0;
  if ( !a1 )
    return v6;
  v7 = a1;
  if ( *(_BYTE *)(a1 + 40) )
    return v6;
  if ( *(_BYTE *)(a1 + 42) )
  {
    if ( !(_DWORD)a2 && word_4F06418[0] == 76 )
      sub_7B8B50(a1, a2, a3, a4, a5, a6);
    return v6;
  }
  v6 = *(_QWORD *)(a1 + 8);
  if ( !(_DWORD)a2 && word_4F06418[0] == 76 )
    sub_867610(a1, a2);
  v9 = *(_QWORD *)(a1 + 16);
  if ( v9 )
  {
    if ( unk_4F04C48 != -1 )
    {
      v10 = qword_4F04C68[0] + 776LL * unk_4F04C48;
      if ( !*(_BYTE *)(a1 + 42) )
      {
        if ( v10 )
        {
          if ( (*(_BYTE *)(v10 + 6) & 0x10) != 0 )
          {
            v11 = *(__int64 **)(v9 + 8);
            if ( v11 )
            {
              v6 = 0;
              while ( 1 )
              {
                if ( !*((_DWORD *)v11 + 8) )
                {
                  v12 = v11[10];
                  if ( v12 )
                  {
                    v13 = *(_QWORD *)(v12 + 16);
                    if ( v13 )
                    {
                      if ( v6 )
                        goto LABEL_24;
                      v6 = v13;
                    }
                  }
                }
                v11 = (__int64 *)*v11;
                if ( !v11 )
                  goto LABEL_25;
              }
            }
          }
        }
      }
    }
    goto LABEL_24;
  }
  v15 = *(_BYTE *)(v6 + 60);
  *(_DWORD *)(v6 + 20) = dword_4F06650[0];
  if ( !v15 && *(_QWORD *)a1 )
  {
    if ( *(_BYTE *)(a1 + 52) )
    {
      *(_BYTE *)(*(_QWORD *)a1 + 52LL) = 1;
      goto LABEL_31;
    }
    if ( *(_QWORD *)(v6 + 24) )
      goto LABEL_24;
LABEL_74:
    *(_QWORD *)(v6 + 8) = 0;
    *(_QWORD *)v6 = qword_4F5FD38;
    qword_4F5FD38 = v6;
    v6 = 0;
    *(_QWORD *)(v7 + 8) = 0;
    return v6;
  }
  *(_BYTE *)(v6 + 62) = 1;
  a2 = (unsigned int *)(sub_85B130(a1, a2, 0, a4, a5) + 664);
  v22 = *(unsigned int **)a2;
  if ( *(_QWORD *)a2 )
  {
    a4 = *(unsigned int *)(v6 + 16);
    while ( 1 )
    {
      v9 = v22[7];
      if ( (unsigned int)v9 >= (unsigned int)a4 )
        break;
      a2 = v22;
      v22 = *(unsigned int **)v22;
      if ( !v22 )
        goto LABEL_32;
    }
    a1 = 0;
    while ( v21 >= (unsigned int)v9 )
    {
      a4 = *(_QWORD *)v22;
      if ( !*(_QWORD *)v22 )
        goto LABEL_51;
      v9 = *(unsigned int *)(a4 + 28);
      a1 = (unsigned __int64)v22;
      v22 = *(unsigned int **)v22;
    }
    if ( !a1 )
      goto LABEL_32;
    v22 = (unsigned int *)a1;
LABEL_51:
    *(_QWORD *)(v6 + 24) = *(_QWORD *)a2;
    *(_QWORD *)a2 = *(_QWORD *)v22;
    *(_QWORD *)v22 = 0;
    v9 = *(_QWORD *)(v6 + 24);
    if ( !v9 )
    {
      v16 = *(_BYTE *)(v6 + 60);
      goto LABEL_78;
    }
    if ( *(_BYTE *)(v9 + 96) )
      goto LABEL_62;
LABEL_53:
    *(_BYTE *)(v6 + 62) = 0;
LABEL_54:
    v23 = *(unsigned int **)v9;
    if ( *(_QWORD *)v9 )
    {
      a4 = v9;
      do
      {
        while ( 1 )
        {
          a1 = *((_QWORD *)v23 + 1);
          if ( *(_QWORD *)(v9 + 8) == a1 )
            break;
          a4 = (__int64)v23;
          v23 = *(unsigned int **)v23;
          if ( !v23 )
            goto LABEL_59;
        }
        a2 = *(unsigned int **)v23;
        *(_QWORD *)a4 = *(_QWORD *)v23;
        v23 = *(unsigned int **)v23;
      }
      while ( v23 );
LABEL_59:
      if ( *(_DWORD *)(v9 + 32) != 1 )
        goto LABEL_60;
    }
    else if ( *(_DWORD *)(v9 + 32) != 1 )
    {
      goto LABEL_31;
    }
    a4 = qword_4F04C68[0];
    v24 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( v24 )
    {
      a2 = 0;
      while ( 1 )
      {
        if ( *(_BYTE *)(v24 + 4) == 17 )
        {
          a1 = *(unsigned int *)(*(_QWORD *)(v9 + 8) + 40LL);
          if ( *(_DWORD *)v24 == (_DWORD)a1 )
          {
            *(_DWORD *)(v9 + 48) = (_DWORD)a2;
LABEL_60:
            v9 = *(_QWORD *)v9;
            if ( v9 )
            {
              if ( !*(_BYTE *)(v9 + 96) )
                goto LABEL_53;
LABEL_62:
              *(_BYTE *)(v6 + 63) = 1;
              goto LABEL_54;
            }
            break;
          }
          a2 = (unsigned int *)(unsigned int)((_DWORD)a2 + 1);
        }
        v25 = *(int *)(v24 + 552);
        if ( (_DWORD)v25 != -1 )
        {
          v24 = qword_4F04C68[0] + 776 * v25;
          if ( v24 )
            continue;
        }
        break;
      }
    }
LABEL_31:
    v16 = *(_BYTE *)(v6 + 60);
  }
LABEL_32:
  if ( !*(_QWORD *)(v6 + 24) )
  {
LABEL_78:
    if ( v16 && !*(_BYTE *)(v7 + 43) )
      sub_6851C0(0x780u, (_DWORD *)(v6 + 32));
    goto LABEL_74;
  }
  if ( !v16 )
  {
LABEL_24:
    v6 = 0;
    goto LABEL_25;
  }
  v17 = sub_85B130(a1, a2, v9, a4, a5);
  v18 = *(_QWORD *)(v17 + 408);
  *(_QWORD *)(v17 + 648) = v6;
  v19 = *(__int64 **)(v18 + 80);
  if ( v19 )
  {
    while ( *((_DWORD *)v19 + 4) > *(_DWORD *)(v6 + 16) )
    {
      v19 = (__int64 *)v19[1];
      if ( !v19 )
        goto LABEL_81;
    }
    v20 = *v19;
    if ( *v19 )
    {
      *(_QWORD *)(v20 + 8) = v6;
      v20 = *v19;
    }
    *(_QWORD *)v6 = v20;
    *(_QWORD *)(v6 + 8) = v19;
    *v19 = v6;
  }
  else
  {
LABEL_81:
    v26 = *(_QWORD *)(v18 + 72);
    *(_QWORD *)v6 = v26;
    if ( v26 )
      *(_QWORD *)(v26 + 8) = v6;
    *(_QWORD *)(v18 + 72) = v6;
  }
  if ( !*(_QWORD *)v6 )
    *(_QWORD *)(v18 + 80) = v6;
LABEL_25:
  v14 = *(_QWORD *)(v7 + 8);
  if ( *(_BYTE *)(v14 + 60) )
    return v6;
  sub_85B940(*(__int64 **)(v14 + 24));
  return v6;
}
