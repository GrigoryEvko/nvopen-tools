// Function: sub_1339310
// Address: 0x1339310
//
__int64 __fastcall sub_1339310(
        struct __pthread_internal_list *a1,
        __int64 a2,
        __int64 a3,
        _BYTE *a4,
        _BOOL8 *a5,
        unsigned __int8 *a6,
        __int64 a7)
{
  unsigned int v9; // r15d
  unsigned __int8 v11; // al
  _BOOL8 v12; // rdx
  unsigned __int8 v13; // dl
  _BOOL8 v14; // rdx
  _BOOL8 v15; // r8
  _BOOL4 v16; // edi
  unsigned int v17; // eax
  __int64 v18; // rdx
  _BOOL4 v19; // edi
  unsigned int v20; // eax
  __int64 v21; // rdx
  _BYTE v23[49]; // [rsp+1Fh] [rbp-31h]

  sub_131ADF0();
  if ( pthread_mutex_trylock(&stru_4F96C00) )
  {
    sub_130AD90((__int64)&xmmword_4F96BC0);
    byte_4F96C28 = 1;
  }
  ++*((_QWORD *)&xmmword_4F96BF0 + 1);
  if ( a1 != (struct __pthread_internal_list *)xmmword_4F96BF0 )
  {
    ++qword_4F96BE8;
    *(_QWORD *)&xmmword_4F96BF0 = a1;
  }
  if ( pthread_mutex_trylock(&stru_5260DA0) )
  {
    sub_130AD90((__int64)qword_5260D60);
    unk_5260DC8 = 1;
  }
  ++unk_5260D98;
  if ( a1 != (struct __pthread_internal_list *)unk_5260D90 )
  {
    ++unk_5260D88;
    unk_5260D90 = a1;
  }
  if ( !a6 )
  {
    v23[0] = byte_5260DD0[0];
    if ( !a4 || !a5 )
      goto LABEL_19;
    v14 = *a5;
    if ( *a5 )
    {
      *a4 = byte_5260DD0[0];
      goto LABEL_19;
    }
    v15 = v14;
    v19 = v14;
    if ( v14 )
    {
      v20 = 0;
      do
      {
        v21 = v20++;
        a4[v21] = v23[v21];
      }
      while ( v20 < v19 );
    }
    goto LABEL_29;
  }
  v9 = 22;
  if ( a7 == 1 )
  {
    v11 = byte_5260DD0[0];
    v23[0] = byte_5260DD0[0];
    if ( !a4 || (a2 = (__int64)a5) == 0 )
    {
LABEL_16:
      v13 = *a6;
      if ( v11 != *a6 )
      {
        byte_5260DD0[0] = *a6;
        if ( v13 )
        {
          if ( !(unsigned __int8)sub_131A4E0(a1, a2) )
            goto LABEL_19;
        }
        else if ( !(unsigned __int8)sub_131A720(a1) )
        {
          goto LABEL_19;
        }
        v9 = 14;
        goto LABEL_11;
      }
LABEL_19:
      v9 = 0;
      goto LABEL_11;
    }
    v12 = *a5;
    if ( *a5 )
    {
      *a4 = byte_5260DD0[0];
      goto LABEL_16;
    }
    v15 = v12;
    v16 = v12;
    if ( v12 )
    {
      v17 = 0;
      do
      {
        v18 = v17++;
        a4[v18] = v23[v18];
      }
      while ( v17 < v16 );
    }
LABEL_29:
    v9 = 22;
    *a5 = v15;
  }
LABEL_11:
  unk_5260DC8 = 0;
  pthread_mutex_unlock(&stru_5260DA0);
  byte_4F96C28 = 0;
  pthread_mutex_unlock(&stru_4F96C00);
  return v9;
}
