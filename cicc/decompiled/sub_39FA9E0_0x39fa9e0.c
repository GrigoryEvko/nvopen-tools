// Function: sub_39FA9E0
// Address: 0x39fa9e0
//
unsigned int *__fastcall sub_39FA9E0(unsigned __int64 a1, _QWORD *a2)
{
  unsigned int *v4; // r15
  __int64 v6; // rbx
  __int16 v7; // di
  char v8; // al
  unsigned __int8 v9; // dl
  char *v10; // rsi
  __int64 v11; // rdi
  unsigned int *v12; // rax
  __int64 *v13; // rdx
  _QWORD *v14; // rax
  unsigned __int64 v15; // [rsp+0h] [rbp-68h] BYREF
  __int64 v16; // [rsp+8h] [rbp-60h]
  __int64 v17; // [rsp+10h] [rbp-58h]
  __int64 v18; // [rsp+18h] [rbp-50h]
  unsigned int *v19; // [rsp+20h] [rbp-48h]
  int v20; // [rsp+28h] [rbp-40h]

  if ( dword_50578E8 )
  {
    if ( &_pthread_key_create )
      pthread_mutex_lock(&stru_50578C0);
    v6 = qword_50578F0;
    if ( !qword_50578F0 )
      goto LABEL_23;
    while ( a1 < *(_QWORD *)v6 )
    {
      v6 = *(_QWORD *)(v6 + 40);
      if ( !v6 )
        goto LABEL_23;
    }
    v4 = sub_39F9D50(v6, a1);
    if ( !v4 )
    {
LABEL_23:
      while ( 1 )
      {
        v6 = qword_50578F8;
        if ( !qword_50578F8 )
          break;
        v11 = qword_50578F8;
        qword_50578F8 = *(_QWORD *)(qword_50578F8 + 40);
        v12 = sub_39F9D50(v11, a1);
        v13 = &qword_50578F0;
        v4 = v12;
        v14 = (_QWORD *)qword_50578F0;
        if ( qword_50578F0 )
        {
          do
          {
            if ( *v14 < *(_QWORD *)v6 )
              break;
            v13 = v14 + 5;
            v14 = (_QWORD *)v14[5];
          }
          while ( v14 );
        }
        *(_QWORD *)(v6 + 40) = v14;
        *v13 = v6;
        if ( v4 )
          goto LABEL_13;
      }
      if ( &_pthread_key_create )
        pthread_mutex_unlock(&stru_50578C0);
      goto LABEL_2;
    }
LABEL_13:
    if ( &_pthread_key_create )
      pthread_mutex_unlock(&stru_50578C0);
    *a2 = *(_QWORD *)(v6 + 8);
    a2[1] = *(_QWORD *)(v6 + 16);
    if ( (*(_BYTE *)(v6 + 32) & 4) != 0 )
    {
      v8 = sub_39F8CF0((__int64)v4 - (int)v4[1] + 4);
      LOBYTE(v7) = v8;
    }
    else
    {
      v7 = *(_WORD *)(v6 + 32) >> 3;
      v8 = v7;
    }
    if ( v8 != -1 )
    {
      v9 = v8 & 0x70;
      if ( (v8 & 0x70) == 0x20 )
      {
        v10 = *(char **)(v6 + 8);
        goto LABEL_22;
      }
      if ( v9 <= 0x20u )
      {
        if ( (v8 & 0x60) != 0 )
          goto LABEL_39;
      }
      else
      {
        if ( v9 == 48 )
        {
          v10 = *(char **)(v6 + 16);
LABEL_22:
          sub_39F8BA0(v7, v10, (char *)v4 + 8, &v15);
          a2[2] = v15;
          return v4;
        }
        if ( v9 != 80 )
LABEL_39:
          abort();
      }
    }
    v10 = 0;
    goto LABEL_22;
  }
LABEL_2:
  v15 = a1;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v20 = 1;
  if ( dl_iterate_phdr((int (*)(struct dl_phdr_info *, size_t, void *))callback, &v15) < 0 )
    return 0;
  v4 = v19;
  if ( v19 )
  {
    *a2 = v16;
    a2[1] = v17;
    a2[2] = v18;
  }
  return v4;
}
