// Function: sub_CEECD0
// Address: 0xceecd0
//
__int64 __fastcall sub_CEECD0(__int64 a1, unsigned __int64 a2)
{
  __int64 v2; // rbx
  unsigned int v3; // eax
  unsigned __int64 v4; // rbx
  char v5; // cl
  char v6; // r8
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // r13

  v2 = a2;
  if ( !byte_4F86488 && (unsigned int)sub_2207590(&byte_4F86488) )
  {
    qword_4F864A0 = 0;
    qword_4F864B0 = (__int64)&unk_4F864C0;
    qword_4F864B8 = 0x400000000LL;
    qword_4F864A8 = 0;
    qword_4F864E0 = (__int64)&qword_4F864F0;
    qword_4F864E8 = 0;
    qword_4F864F0 = 0;
    qword_4F864F8 = 1;
    __cxa_atexit((void (*)(void *))sub_CEEBF0, &qword_4F864F0 - 10, &qword_4A427C0);
    sub_2207640(&byte_4F86488);
  }
  if ( &_pthread_key_create )
  {
    v3 = pthread_mutex_lock(&stru_4F86500);
    if ( v3 )
      sub_4264C5(v3);
  }
  if ( a2 )
  {
    _BitScanReverse64(&v4, a2);
    v5 = 63 - (v4 ^ 0x3F);
    v2 = 1LL << v5;
    v6 = v5;
    v7 = -(1LL << v5);
  }
  else
  {
    v7 = 0;
    v6 = -1;
  }
  qword_4F864F0 += a1;
  v8 = v7 & (qword_4F864A0 + v2 - 1);
  if ( qword_4F864A8 >= (unsigned __int64)(a1 + v8) && qword_4F864A0 )
  {
    qword_4F864A0 = a1 + v8;
    v9 = v8;
  }
  else
  {
    v9 = sub_9D1E70((__int64)&qword_4F864A0, a1, a1, v6);
  }
  if ( &_pthread_key_create )
    pthread_mutex_unlock(&stru_4F86500);
  return v9;
}
