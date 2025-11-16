// Function: sub_1339B40
// Address: 0x1339b40
//
__int64 __fastcall sub_1339B40(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        unsigned __int64 *a5,
        __int64 *a6,
        __int64 a7)
{
  __int128 *v10; // rdi
  __int64 v11; // rdx
  unsigned __int64 v12; // rbx
  __int64 v13; // rsi
  unsigned int v14; // r15d
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  __int64 v17; // rax
  unsigned __int64 v19; // r8
  char *v20; // r13
  unsigned int v21; // ecx
  __int64 v22; // rdi
  unsigned __int64 v23; // r8
  char *v24; // r13
  unsigned int v25; // eax
  __int64 v26; // rdi
  unsigned __int64 v27; // r8
  char *v28; // r13
  unsigned int v29; // ecx
  __int64 v30; // rdi
  _BYTE v32[8]; // [rsp+10h] [rbp-50h]
  __int64 (__fastcall **v33)(int, int, int, int, int, int, int); // [rsp+18h] [rbp-48h] BYREF
  _QWORD v34[8]; // [rsp+20h] [rbp-40h] BYREF

  v10 = (__int128 *)&stru_4F96C00;
  if ( pthread_mutex_trylock(&stru_4F96C00) )
  {
    v10 = &xmmword_4F96BC0;
    sub_130AD90((__int64)&xmmword_4F96BC0);
    byte_4F96C28 = 1;
  }
  ++*((_QWORD *)&xmmword_4F96BF0 + 1);
  if ( a1 != (_QWORD)xmmword_4F96BF0 )
  {
    ++qword_4F96BE8;
    *(_QWORD *)&xmmword_4F96BF0 = a1;
  }
  v12 = *(_QWORD *)(a2 + 8);
  if ( v12 > 0xFFFFFFFF || (unsigned int)sub_1300B70(v10, a2, v11) <= (unsigned int)v12 )
    goto LABEL_23;
  v13 = qword_50579C0[(unsigned int)v12];
  if ( !v13 )
  {
    v14 = 14;
    if ( unk_505F9B8 <= (unsigned int)v12 )
      goto LABEL_24;
    v33 = &off_49E8020;
    if ( a4 && a5 )
    {
      v16 = *a5;
      if ( *a5 != 8 )
      {
        if ( v16 > 8 )
          v16 = 8;
        if ( (unsigned int)v16 >= 8 )
        {
          *a4 = (__int64)&off_49E8020;
          v23 = (unsigned __int64)(a4 + 1) & 0xFFFFFFFFFFFFFFF8LL;
          *(__int64 *)((char *)a4 + (unsigned int)v16 - 8) = *(_QWORD *)&v32[(unsigned int)v16];
          v24 = (char *)a4 - v23;
          if ( (((_DWORD)v16 + (_DWORD)v24) & 0xFFFFFFF8) >= 8 )
          {
            v25 = 0;
            do
            {
              v26 = v25;
              v25 += 8;
              *(_QWORD *)(v23 + v26) = *(_QWORD *)((char *)&v33 - v24 + v26);
            }
            while ( v25 < (((_DWORD)v16 + (_DWORD)v24) & 0xFFFFFFF8) );
          }
          goto LABEL_34;
        }
        goto LABEL_35;
      }
      *a4 = (__int64)&off_49E8020;
    }
    if ( !a6 )
      goto LABEL_14;
    v14 = 22;
    if ( a7 != 8 )
      goto LABEL_24;
    v17 = *a6;
    v34[1] = _mm_loadu_si128((const __m128i *)&off_49E8000).m128i_i64[1];
    v34[0] = v17;
    if ( sub_1300B80(a1, v12, (__int64)v34) )
      goto LABEL_14;
LABEL_23:
    v14 = 14;
    goto LABEL_24;
  }
  if ( !a6 )
  {
    v15 = *(_QWORD *)(sub_1316370(qword_50579C0[(unsigned int)v12]) + 8);
    v33 = (__int64 (__fastcall **)(int, int, int, int, int, int, int))v15;
    if ( !a4 || !a5 )
      goto LABEL_14;
    v16 = *a5;
    if ( *a5 == 8 )
      goto LABEL_13;
    if ( v16 > 8 )
      v16 = 8;
    if ( (unsigned int)v16 >= 8 )
    {
      *a4 = v15;
      v19 = (unsigned __int64)(a4 + 1) & 0xFFFFFFFFFFFFFFF8LL;
      *(__int64 *)((char *)a4 + (unsigned int)v16 - 8) = *(_QWORD *)&v32[(unsigned int)v16];
      v20 = (char *)a4 - v19;
      if ( (((_DWORD)v16 + (_DWORD)v20) & 0xFFFFFFF8) >= 8 )
      {
        v21 = 0;
        do
        {
          v22 = v21;
          v21 += 8;
          *(_QWORD *)(v19 + v22) = *(_QWORD *)((char *)&v33 - v20 + v22);
        }
        while ( v21 < (((_DWORD)v16 + (_DWORD)v20) & 0xFFFFFFF8) );
      }
      goto LABEL_34;
    }
    goto LABEL_35;
  }
  v14 = 22;
  if ( a7 == 8 )
  {
    v15 = sub_1316E90(a1, v13, *a6);
    v33 = (__int64 (__fastcall **)(int, int, int, int, int, int, int))v15;
    if ( !a4 || !a5 )
      goto LABEL_14;
    v16 = *a5;
    if ( *a5 == 8 )
    {
LABEL_13:
      *a4 = v15;
LABEL_14:
      v14 = 0;
      goto LABEL_24;
    }
    if ( v16 > 8 )
      v16 = 8;
    if ( (unsigned int)v16 >= 8 )
    {
      *a4 = v15;
      v27 = (unsigned __int64)(a4 + 1) & 0xFFFFFFFFFFFFFFF8LL;
      *(__int64 *)((char *)a4 + (unsigned int)v16 - 8) = *(_QWORD *)&v32[(unsigned int)v16];
      v28 = (char *)a4 - v27;
      if ( (((_DWORD)v16 + (_DWORD)v28) & 0xFFFFFFF8) >= 8 )
      {
        v29 = 0;
        do
        {
          v30 = v29;
          v29 += 8;
          *(_QWORD *)(v27 + v30) = *(_QWORD *)((char *)&v33 - v28 + v30);
        }
        while ( v29 < (((_DWORD)v16 + (_DWORD)v28) & 0xFFFFFFF8) );
      }
LABEL_34:
      v14 = 22;
      *a5 = v16;
      goto LABEL_24;
    }
LABEL_35:
    if ( (v16 & 4) != 0 )
    {
      *(_DWORD *)a4 = (_DWORD)v33;
      *(_DWORD *)((char *)a4 + (unsigned int)v16 - 4) = *(_DWORD *)&v32[(unsigned int)v16 + 4];
    }
    else if ( (_DWORD)v16 )
    {
      *(_BYTE *)a4 = (_BYTE)v33;
      if ( (v16 & 2) != 0 )
        *(_WORD *)((char *)a4 + (unsigned int)v16 - 2) = *(_WORD *)&v32[(unsigned int)v16 + 6];
    }
    goto LABEL_34;
  }
LABEL_24:
  byte_4F96C28 = 0;
  pthread_mutex_unlock(&stru_4F96C00);
  return v14;
}
