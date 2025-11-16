// Function: sub_1339820
// Address: 0x1339820
//
__int64 __fastcall sub_1339820(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        _QWORD *a4,
        unsigned __int64 *a5,
        unsigned __int64 *a6,
        __int64 a7)
{
  unsigned int v7; // r15d
  __int128 *v9; // rdi
  __int64 v12; // rdx
  unsigned __int64 v13; // rbx
  __int64 v15; // rsi
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rdi
  unsigned int v19; // edx
  __int64 v20; // rsi
  unsigned __int64 *v21; // [rsp+8h] [rbp-48h]
  __int64 v22; // [rsp+10h] [rbp-40h] BYREF
  unsigned __int64 v23; // [rsp+18h] [rbp-38h] BYREF

  v7 = 2;
  v21 = a5;
  if ( unk_4C6F2C8 )
  {
    v9 = (__int128 *)&stru_4F96C00;
    if ( pthread_mutex_trylock(&stru_4F96C00) )
    {
      v9 = &xmmword_4F96BC0;
      sub_130AD90((__int64)&xmmword_4F96BC0);
      byte_4F96C28 = 1;
    }
    ++*((_QWORD *)&xmmword_4F96BF0 + 1);
    if ( a1 != (_QWORD)xmmword_4F96BF0 )
    {
      ++qword_4F96BE8;
      *(_QWORD *)&xmmword_4F96BF0 = a1;
    }
    v13 = *(_QWORD *)(a2 + 8);
    if ( v13 <= 0xFFFFFFFF
      && (unsigned int)sub_1300B70(v9, a2, v12) > (unsigned int)v13
      && (v15 = qword_50579C0[(unsigned int)v13]) != 0 )
    {
      if ( a6 )
      {
        if ( a7 != 8 )
          goto LABEL_28;
        v16 = *a6;
        a6 = &v23;
        v23 = v16;
      }
      v7 = 14;
      if ( !(unsigned __int8)sub_1317070(a1, v15, &v22, a6) )
      {
        if ( !a4 || !v21 )
        {
          v7 = 0;
          goto LABEL_8;
        }
        v17 = *v21;
        if ( *v21 == 8 )
        {
          v7 = 0;
          *a4 = v22;
          goto LABEL_8;
        }
        if ( v17 > 8 )
          v17 = 8;
        if ( (unsigned int)v17 >= 8 )
        {
          v18 = (unsigned __int64)(a4 + 1) & 0xFFFFFFFFFFFFFFF8LL;
          *a4 = v22;
          *(_QWORD *)((char *)a4 + (unsigned int)v17 - 8) = *(unsigned __int64 **)((char *)&v21 + (unsigned int)v17);
          if ( (((_DWORD)v17 + (_DWORD)a4 - (_DWORD)v18) & 0xFFFFFFF8) >= 8 )
          {
            v19 = 0;
            do
            {
              v20 = v19;
              v19 += 8;
              *(_QWORD *)(v18 + v20) = *(_QWORD *)((char *)&v22 - ((char *)a4 - v18) + v20);
            }
            while ( v19 < (((_DWORD)v17 + (_DWORD)a4 - (_DWORD)v18) & 0xFFFFFFF8) );
          }
        }
        else if ( (v17 & 4) != 0 )
        {
          *(_DWORD *)a4 = v22;
          *(_DWORD *)((char *)a4 + (unsigned int)v17 - 4) = *(_DWORD *)((char *)&v21 + (unsigned int)v17 + 4);
        }
        else if ( (_DWORD)v17 )
        {
          *(_BYTE *)a4 = v22;
          if ( (v17 & 2) != 0 )
            *(_WORD *)((char *)a4 + (unsigned int)v17 - 2) = *(_WORD *)((char *)&v21 + (unsigned int)v17 + 6);
        }
        *v21 = v17;
LABEL_28:
        v7 = 22;
      }
    }
    else
    {
      v7 = 14;
    }
LABEL_8:
    byte_4F96C28 = 0;
    pthread_mutex_unlock(&stru_4F96C00);
  }
  return v7;
}
