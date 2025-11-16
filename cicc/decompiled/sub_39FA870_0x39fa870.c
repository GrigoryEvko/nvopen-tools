// Function: sub_39FA870
// Address: 0x39fa870
//
__int64 __fastcall sub_39FA870(_DWORD *a1)
{
  __int64 v2; // rax
  __int64 *v3; // rcx
  _DWORD *v4; // rdx
  __int64 v5; // r12
  __int64 *v7; // rax
  _QWORD *v8; // rdi

  if ( !a1 || !*a1 )
    return 0;
  if ( &_pthread_key_create )
    pthread_mutex_lock(&stru_50578C0);
  v2 = qword_50578F8;
  if ( qword_50578F8 )
  {
    v3 = &qword_50578F8;
    do
    {
      v4 = *(_DWORD **)(v2 + 24);
      v5 = v2;
      v2 = *(_QWORD *)(v2 + 40);
      if ( a1 == v4 )
      {
        *v3 = v2;
        goto LABEL_10;
      }
      v3 = (__int64 *)(v5 + 40);
    }
    while ( v2 );
  }
  v5 = qword_50578F0;
  if ( !qword_50578F0 )
  {
LABEL_21:
    if ( &_pthread_key_create )
      pthread_mutex_unlock(&stru_50578C0);
    abort();
  }
  v7 = &qword_50578F0;
  while ( 1 )
  {
    v8 = *(_QWORD **)(v5 + 24);
    if ( (*(_BYTE *)(v5 + 32) & 1) != 0 )
      break;
    if ( a1 == (_DWORD *)v8 )
    {
      *v7 = *(_QWORD *)(v5 + 40);
      goto LABEL_10;
    }
LABEL_16:
    v7 = (__int64 *)(v5 + 40);
    v5 = *(_QWORD *)(v5 + 40);
    if ( !v5 )
      goto LABEL_21;
  }
  if ( a1 != (_DWORD *)*v8 )
    goto LABEL_16;
  *v7 = *(_QWORD *)(v5 + 40);
  _libc_free((unsigned __int64)v8);
LABEL_10:
  if ( &_pthread_key_create )
    pthread_mutex_unlock(&stru_50578C0);
  return v5;
}
