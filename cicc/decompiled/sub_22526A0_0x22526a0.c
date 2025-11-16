// Function: sub_22526A0
// Address: 0x22526a0
//
unsigned __int64 *__fastcall sub_22526A0(unsigned __int64 a1)
{
  __int64 v1; // rcx
  unsigned __int64 *v2; // r12
  unsigned __int64 v3; // rcx
  __int64 *v4; // rsi
  unsigned __int64 v5; // rdx
  _QWORD *v6; // rax
  unsigned __int64 v7; // rdi
  unsigned __int64 *v9; // rdx

  if ( &_pthread_key_create && pthread_mutex_lock(&stru_4FD6AA0) )
    JUMPOUT(0x4265C8);
  v1 = a1 + 16;
  v2 = (unsigned __int64 *)qword_4FD6AC8;
  if ( a1 >= 0xFFFFFFFFFFFFFFF0LL )
    v1 = 16;
  v3 = (v1 + 15) & 0xFFFFFFFFFFFFFFF0LL;
  if ( qword_4FD6AC8 )
  {
    v4 = &qword_4FD6AC8;
    while ( 1 )
    {
      v5 = *v2;
      v6 = v2;
      v2 = (unsigned __int64 *)v2[1];
      if ( v3 <= v5 )
        break;
      v4 = v6 + 1;
      if ( !v2 )
        goto LABEL_12;
    }
    v7 = v5 - v3;
    if ( v5 - v3 > 0xF )
    {
      v9 = (_QWORD *)((char *)v6 + v3);
      v9[1] = (unsigned __int64)v2;
      v6 = (_QWORD *)*v4;
      *v9 = v7;
      *v6 = v3;
      *v4 = (__int64)v9;
    }
    else
    {
      *v6 = v5;
      *v4 = (__int64)v2;
    }
    v2 = v6 + 2;
  }
LABEL_12:
  if ( &_pthread_key_create && pthread_mutex_unlock(&stru_4FD6AA0) )
    JUMPOUT(0x4265CD);
  return v2;
}
