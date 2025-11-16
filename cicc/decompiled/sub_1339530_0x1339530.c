// Function: sub_1339530
// Address: 0x1339530
//
__int64 __fastcall sub_1339530(
        struct __pthread_internal_list *a1,
        __int64 a2,
        __int64 a3,
        _QWORD *a4,
        __int64 *a5,
        unsigned __int64 *a6,
        __int64 a7)
{
  signed __int64 v10; // rax
  unsigned __int64 v11; // r13
  unsigned int v12; // r12d
  __int64 v13; // rdx
  unsigned __int64 v14; // rdi
  char *v15; // r13
  char *v16; // rcx
  unsigned int v17; // r13d
  unsigned int v18; // r13d
  unsigned int v19; // eax
  __int64 v20; // rsi
  unsigned __int64 v22; // r8
  char *v23; // r13
  unsigned int v24; // ecx
  __int64 v25; // rdi
  _QWORD v26[7]; // [rsp+8h] [rbp-38h] BYREF

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
    v26[0] = qword_5260D48[0];
    if ( !a4 || !a5 )
      goto LABEL_19;
    v13 = *a5;
    if ( *a5 == 8 )
    {
      *a4 = qword_5260D48[0];
      v12 = 0;
      goto LABEL_28;
    }
    if ( (unsigned __int64)*a5 > 8 )
      v13 = 8;
    if ( (unsigned int)v13 >= 8 )
    {
      *a4 = qword_5260D48[0];
      v22 = (unsigned __int64)(a4 + 1) & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)((char *)a4 + (unsigned int)v13 - 8) = *(_QWORD *)((char *)&v26[-1] + (unsigned int)v13);
      v23 = (char *)a4 - v22;
      if ( (((_DWORD)v13 + (_DWORD)v23) & 0xFFFFFFF8) >= 8 )
      {
        v24 = 0;
        do
        {
          v25 = v24;
          v24 += 8;
          *(_QWORD *)(v22 + v25) = *(_QWORD *)((char *)v26 - v23 + v25);
        }
        while ( v24 < (((_DWORD)v13 + (_DWORD)v23) & 0xFFFFFFF8) );
      }
      goto LABEL_26;
    }
    goto LABEL_36;
  }
  if ( a7 != 8 )
    goto LABEL_27;
  v10 = qword_5260D48[0];
  v26[0] = qword_5260D48[0];
  if ( !a4 || !a5 )
    goto LABEL_15;
  if ( *a5 != 8 )
  {
    v13 = *a5;
    if ( (unsigned __int64)*a5 > 8 )
      v13 = 8;
    if ( (unsigned int)v13 >= 8 )
    {
      *a4 = qword_5260D48[0];
      v14 = (unsigned __int64)(a4 + 1) & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)((char *)a4 + (unsigned int)v13 - 8) = *(_QWORD *)((char *)&v26[-1] + (unsigned int)v13);
      v15 = (char *)a4 - v14;
      v16 = (char *)((char *)v26 - v15);
      v17 = (v13 + (_DWORD)v15) & 0xFFFFFFF8;
      if ( v17 >= 8 )
      {
        v18 = v17 & 0xFFFFFFF8;
        v19 = 0;
        do
        {
          v20 = v19;
          v19 += 8;
          *(_QWORD *)(v14 + v20) = *(_QWORD *)&v16[v20];
        }
        while ( v19 < v18 );
      }
LABEL_26:
      *a5 = v13;
      goto LABEL_27;
    }
LABEL_36:
    if ( (v13 & 4) != 0 )
    {
      *(_DWORD *)a4 = v26[0];
      *(_DWORD *)((char *)a4 + (unsigned int)v13 - 4) = *(_DWORD *)((char *)v26 + (unsigned int)v13 - 4);
    }
    else if ( (_DWORD)v13 )
    {
      *(_BYTE *)a4 = v26[0];
      if ( (v13 & 2) != 0 )
        *(_WORD *)((char *)a4 + (unsigned int)v13 - 2) = *(_WORD *)((char *)v26 + (unsigned int)v13 - 2);
    }
    goto LABEL_26;
  }
  *a4 = qword_5260D48[0];
LABEL_15:
  v11 = *a6;
  if ( v10 != *a6 )
  {
    if ( unk_4C6F268 >= v11 )
    {
      if ( byte_5260DD0[0] )
      {
        byte_5260DD0[0] = 0;
        if ( (unsigned __int8)sub_131A720(a1)
          || (qword_5260D48[0] = v11, byte_5260DD0[0] = 1, (unsigned __int8)sub_131A4E0(a1, a2)) )
        {
          v12 = 14;
          goto LABEL_28;
        }
      }
      else
      {
        qword_5260D48[0] = *a6;
      }
      goto LABEL_19;
    }
LABEL_27:
    v12 = 22;
    goto LABEL_28;
  }
LABEL_19:
  v12 = 0;
LABEL_28:
  unk_5260DC8 = 0;
  pthread_mutex_unlock(&stru_5260DA0);
  byte_4F96C28 = 0;
  pthread_mutex_unlock(&stru_4F96C00);
  return v12;
}
