// Function: sub_85FCF0
// Address: 0x85fcf0
//
void sub_85FCF0()
{
  __int64 v0; // r13
  _QWORD *v1; // rcx
  __int64 v2; // rsi
  _QWORD *v3; // rax
  _QWORD *v4; // rdx
  __int64 *v5; // rdi
  __int64 v6; // rbx
  int v7; // r12d
  __int64 v8; // rbx
  __int64 v9; // rdi
  char v10; // [rsp+Fh] [rbp-31h]

  v0 = (__int64)qword_4F04C18;
  v1 = (_QWORD *)qword_4F04C18[2];
  qword_4F04C18 = (_QWORD *)*qword_4F04C18;
  if ( v1 )
  {
    v2 = v1[1];
    if ( v2 )
    {
      v3 = (_QWORD *)v1[1];
      do
      {
        v4 = v3;
        v3 = (_QWORD *)*v3;
      }
      while ( v3 );
      *v4 = qword_4F5FD48;
      qword_4F5FD48 = v2;
    }
    *v1 = qword_4F5FD30;
    qword_4F5FD30 = (__int64)v1;
  }
  v5 = *(__int64 **)(v0 + 32);
  if ( v5 )
    sub_725130(v5);
  if ( *(_QWORD *)(v0 + 8) )
  {
    if ( unk_4F04C48 != -1 )
    {
      v6 = qword_4F04C68[0] + 776LL * unk_4F04C48;
      if ( v6 )
      {
        v7 = 1;
        v10 = *(_BYTE *)(v0 + 45);
        while ( 1 )
        {
          if ( *(_BYTE *)(v6 + 4) == 9 && (*(_BYTE *)(v6 + 7) & 1) != 0 )
          {
            sub_85BC50(**(_QWORD ***)(v6 + 408), *(_QWORD *)(v6 + 376));
            if ( v7 )
              v9 = (__int64)qword_4F04C18;
            else
              v9 = *(_QWORD *)(v6 + 672);
            if ( v9 && *(_QWORD *)(v9 + 16) )
              sub_85BF70(v9);
            if ( !v10 )
              break;
            v7 = 0;
          }
          v8 = *(int *)(v6 + 552);
          if ( (_DWORD)v8 != -1 )
          {
            v6 = qword_4F04C68[0] + 776 * v8;
            if ( v6 )
              continue;
          }
          break;
        }
      }
    }
  }
  *(_QWORD *)v0 = qword_4F5FD40;
  qword_4F5FD40 = v0;
  if ( qword_4F04C18 && !*((_WORD *)qword_4F04C18 + 20) )
  {
    if ( *((_BYTE *)qword_4F04C18 + 45) )
      sub_85FC80();
  }
}
