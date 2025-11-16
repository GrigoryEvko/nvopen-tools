// Function: sub_1322490
// Address: 0x1322490
//
__int64 __fastcall sub_1322490(
        struct __pthread_internal_list *a1,
        __int64 a2,
        __int64 *a3,
        unsigned __int64 *a4,
        __int64 *a5,
        __int64 a6,
        char a7)
{
  unsigned __int64 v8; // r12
  __int64 result; // rax
  bool v12; // al
  __int64 v13; // rcx
  int v14; // edx
  __int64 v15; // r8
  __int64 v16; // rax
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // rdi
  char *v19; // rbx
  char *v20; // rdx
  unsigned int v21; // ebx
  unsigned int v22; // ebx
  unsigned int v23; // eax
  __int64 v24; // rsi
  __int64 v25; // [rsp+8h] [rbp-58h]
  __int64 v26; // [rsp+10h] [rbp-50h]
  int v27; // [rsp+18h] [rbp-48h]
  __int64 v28; // [rsp+18h] [rbp-48h]
  _BYTE v29[8]; // [rsp+20h] [rbp-40h]
  __int64 v30; // [rsp+28h] [rbp-38h] BYREF

  v8 = *(_QWORD *)(a2 + 8);
  if ( v8 > 0xFFFFFFFF || !qword_50579C0[v8] )
    return 14;
  if ( !a3 || !a4 )
  {
    v26 = qword_50579C0[v8];
    v27 = (a7 == 0) + 1;
    if ( !a5 )
      return 0;
    goto LABEL_6;
  }
  v25 = a6;
  v28 = qword_50579C0[v8];
  v16 = sub_1315150(v28, (unsigned int)(a7 == 0) + 1);
  v17 = *a4;
  a6 = v25;
  v30 = v16;
  if ( v17 == 8 )
  {
    *a3 = v16;
    v26 = v28;
    v27 = (a7 == 0) + 1;
    if ( !a5 )
      return 0;
LABEL_6:
    result = 22;
    if ( a6 != 8 )
      return result;
    v12 = sub_1319070(v8);
    v13 = *a5;
    v14 = v27;
    v15 = v26;
    if ( !v12 || v13 <= 0 )
    {
LABEL_9:
      if ( !(unsigned __int8)sub_1315120((__int64)a1, v15, v14, v13) )
        return 0;
      return 14;
    }
    if ( !(unsigned __int8)sub_131A460(a1, v8) )
    {
      v13 = *a5;
      v15 = v26;
      v14 = v27;
      goto LABEL_9;
    }
    return 14;
  }
  if ( v17 > 8 )
    v17 = 8;
  if ( (unsigned int)v17 >= 8 )
  {
    *a3 = v16;
    v18 = (unsigned __int64)(a3 + 1) & 0xFFFFFFFFFFFFFFF8LL;
    *(__int64 *)((char *)a3 + (unsigned int)v17 - 8) = *(_QWORD *)&v29[(unsigned int)v17];
    v19 = (char *)a3 - v18;
    v20 = (char *)((char *)&v30 - v19);
    v21 = (v17 + (_DWORD)v19) & 0xFFFFFFF8;
    if ( v21 >= 8 )
    {
      v22 = v21 & 0xFFFFFFF8;
      v23 = 0;
      do
      {
        v24 = v23;
        v23 += 8;
        *(_QWORD *)(v18 + v24) = *(_QWORD *)&v20[v24];
      }
      while ( v23 < v22 );
    }
  }
  else if ( (v17 & 4) != 0 )
  {
    *(_DWORD *)a3 = v30;
    *(_DWORD *)((char *)a3 + (unsigned int)v17 - 4) = *(_DWORD *)&v29[(unsigned int)v17 + 4];
  }
  else if ( (_DWORD)v17 )
  {
    *(_BYTE *)a3 = v30;
    if ( (v17 & 2) != 0 )
      *(_WORD *)((char *)a3 + (unsigned int)v17 - 2) = *(_WORD *)&v29[(unsigned int)v17 + 6];
  }
  *a4 = v17;
  return 22;
}
