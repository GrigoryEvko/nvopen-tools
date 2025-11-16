// Function: sub_133A580
// Address: 0x133a580
//
__int64 __fastcall sub_133A580(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        _QWORD *a4,
        unsigned __int64 *a5,
        __int64 a6,
        __int64 a7)
{
  unsigned int v7; // r13d
  __int128 *v11; // rdi
  __int64 v14; // rdx
  unsigned __int64 v15; // r12
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // rax
  unsigned __int64 v19; // r8
  char *v20; // r14
  unsigned int v21; // eax
  __int64 v22; // rdi
  _QWORD v23[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = 22;
  if ( a4 && a5 && *a5 == 8 )
  {
    v11 = (__int128 *)&stru_4F96C00;
    if ( pthread_mutex_trylock(&stru_4F96C00) )
    {
      v11 = &xmmword_4F96BC0;
      sub_130AD90((__int64)&xmmword_4F96BC0);
      byte_4F96C28 = 1;
    }
    ++*((_QWORD *)&xmmword_4F96BF0 + 1);
    if ( a1 != (_QWORD)xmmword_4F96BF0 )
    {
      ++qword_4F96BE8;
      *(_QWORD *)&xmmword_4F96BF0 = a1;
    }
    v7 = 1;
    if ( !(a7 | a6) )
    {
      v15 = *(_QWORD *)(a2 + 16);
      if ( v15 <= 0xFFFFFFFF
        && (unsigned int)sub_1300B70(v11, a2, v14) > (unsigned int)v15
        && (v16 = qword_50579C0[(unsigned int)v15]) != 0 )
      {
        v17 = *a5;
        v18 = v16 + 10656;
        v23[0] = v18;
        if ( v17 == 8 )
        {
          *a4 = v18;
          v7 = 0;
        }
        else
        {
          if ( v17 > 8 )
            v17 = 8;
          if ( (unsigned int)v17 >= 8 )
          {
            *a4 = v18;
            v19 = (unsigned __int64)(a4 + 1) & 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)((char *)a4 + (unsigned int)v17 - 8) = *(_QWORD *)((char *)&v23[-1] + (unsigned int)v17);
            v20 = (char *)a4 - v19;
            if ( (((_DWORD)v17 + (_DWORD)v20) & 0xFFFFFFF8) >= 8 )
            {
              v21 = 0;
              do
              {
                v22 = v21;
                v21 += 8;
                *(_QWORD *)(v19 + v22) = *(_QWORD *)((char *)v23 - v20 + v22);
              }
              while ( v21 < (((_DWORD)v17 + (_DWORD)v20) & 0xFFFFFFF8) );
            }
          }
          else if ( (v17 & 4) != 0 )
          {
            *(_DWORD *)a4 = v23[0];
            *(_DWORD *)((char *)a4 + (unsigned int)v17 - 4) = *(_DWORD *)((char *)v23 + (unsigned int)v17 - 4);
          }
          else if ( (_DWORD)v17 )
          {
            *(_BYTE *)a4 = v23[0];
            if ( (v17 & 2) != 0 )
              *(_WORD *)((char *)a4 + (unsigned int)v17 - 2) = *(_WORD *)((char *)v23 + (unsigned int)v17 - 2);
          }
          *a5 = v17;
          v7 = 22;
        }
      }
      else
      {
        v7 = 14;
      }
    }
    byte_4F96C28 = 0;
    pthread_mutex_unlock(&stru_4F96C00);
  }
  return v7;
}
