// Function: sub_1339050
// Address: 0x1339050
//
__int64 __fastcall sub_1339050(
        __int64 a1,
        char *a2,
        __int64 a3,
        char **a4,
        unsigned __int64 *a5,
        char **a6,
        __int64 a7)
{
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // r15
  __int64 v12; // rbx
  unsigned int v13; // r12d
  unsigned int v15; // eax
  char *v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // r15
  unsigned __int64 v19; // r8
  char *v20; // r13
  unsigned int v21; // ecx
  __int64 v22; // rdi
  __int64 v23; // rdi
  _BYTE v25[8]; // [rsp+10h] [rbp-40h]
  char *s2; // [rsp+18h] [rbp-38h] BYREF

  v10 = (unsigned __int64)&stru_4F96C00;
  s2 = 0;
  if ( pthread_mutex_trylock(&stru_4F96C00) )
  {
    v10 = (unsigned __int64)&xmmword_4F96BC0;
    sub_130AD90((__int64)&xmmword_4F96BC0);
    byte_4F96C28 = 1;
  }
  ++*((_QWORD *)&xmmword_4F96BF0 + 1);
  if ( a1 != (_QWORD)xmmword_4F96BF0 )
  {
    ++qword_4F96BE8;
    *(_QWORD *)&xmmword_4F96BF0 = a1;
  }
  if ( a6 )
  {
    if ( a7 != 8 )
      goto LABEL_13;
    s2 = *a6;
  }
  v11 = *((_QWORD *)a2 + 1);
  if ( v11 > 0xFFFFFFFF )
    goto LABEL_23;
  if ( s2 )
  {
    v12 = 0;
    while ( 1 )
    {
      a2 = s2;
      if ( !strcmp((const char *)*(&off_4C6F2A0 + v12), s2) )
        break;
      if ( ++v12 == 3 )
        goto LABEL_13;
    }
    if ( v11 != 4096 && *(_DWORD *)(qword_4F96BA0 + 8) != (_DWORD)v11 )
    {
      v23 = qword_50579C0[(unsigned int)v11];
      v18 = v23;
      if ( !v23 || (unsigned __int8)sub_1316F70(v23, v12) )
        goto LABEL_23;
      goto LABEL_29;
    }
    v10 = (unsigned int)v12;
    if ( (unsigned __int8)sub_1346410((unsigned int)v12) )
    {
LABEL_23:
      v13 = 14;
      goto LABEL_14;
    }
LABEL_17:
    v15 = sub_1346400(v10, a2);
    goto LABEL_18;
  }
  if ( v11 == 4096 || *(_DWORD *)(qword_4F96BA0 + 8) == (_DWORD)v11 )
    goto LABEL_17;
  if ( !qword_50579C0[(unsigned int)v11] )
    goto LABEL_23;
  v18 = qword_50579C0[(unsigned int)v11];
LABEL_29:
  v15 = sub_1316F60(v18);
LABEL_18:
  v16 = (char *)*(&off_4C6F2A0 + v15);
  s2 = v16;
  if ( !a4 || !a5 )
  {
    v13 = 0;
    goto LABEL_14;
  }
  v17 = *a5;
  if ( *a5 == 8 )
  {
    *a4 = v16;
    v13 = 0;
    goto LABEL_14;
  }
  if ( v17 > 8 )
    v17 = 8;
  if ( (unsigned int)v17 >= 8 )
  {
    *a4 = v16;
    v19 = (unsigned __int64)(a4 + 1) & 0xFFFFFFFFFFFFFFF8LL;
    *(char **)((char *)a4 + (unsigned int)v17 - 8) = *(char **)&v25[(unsigned int)v17];
    v20 = (char *)a4 - v19;
    if ( (((_DWORD)v17 + (_DWORD)v20) & 0xFFFFFFF8) >= 8 )
    {
      v21 = 0;
      do
      {
        v22 = v21;
        v21 += 8;
        *(_QWORD *)(v19 + v22) = *(_QWORD *)((char *)&s2 - v20 + v22);
      }
      while ( v21 < (((_DWORD)v17 + (_DWORD)v20) & 0xFFFFFFF8) );
    }
  }
  else if ( (v17 & 4) != 0 )
  {
    *(_DWORD *)a4 = (_DWORD)s2;
    *(_DWORD *)((char *)a4 + (unsigned int)v17 - 4) = *(_DWORD *)&v25[(unsigned int)v17 + 4];
  }
  else if ( (_DWORD)v17 )
  {
    *(_BYTE *)a4 = (_BYTE)s2;
    if ( (v17 & 2) != 0 )
      *(_WORD *)((char *)a4 + (unsigned int)v17 - 2) = *(_WORD *)&v25[(unsigned int)v17 + 6];
  }
  *a5 = v17;
LABEL_13:
  v13 = 22;
LABEL_14:
  byte_4F96C28 = 0;
  pthread_mutex_unlock(&stru_4F96C00);
  return v13;
}
