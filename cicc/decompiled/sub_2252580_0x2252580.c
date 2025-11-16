// Function: sub_2252580
// Address: 0x2252580
//
int __fastcall sub_2252580(__int64 a1)
{
  __int64 *v2; // rax
  _QWORD *v3; // rdi
  __int64 v4; // r9
  __int64 *v5; // rsi
  __int64 *v6; // rdx
  __int64 *v7; // r8
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // r9

  if ( &_pthread_key_create && pthread_mutex_lock(&stru_4FD6AA0) )
    JUMPOUT(0x4265AC);
  v2 = (__int64 *)qword_4FD6AC8;
  v3 = (_QWORD *)(a1 - 16);
  if ( !qword_4FD6AC8
    || (v4 = *(_QWORD *)(a1 - 16), v5 = (_QWORD *)((char *)v3 + v4), qword_4FD6AC8 > (unsigned __int64)v3 + v4) )
  {
    *(_QWORD *)(a1 - 8) = qword_4FD6AC8;
    qword_4FD6AC8 = a1 - 16;
    goto LABEL_17;
  }
  v6 = *(__int64 **)(qword_4FD6AC8 + 8);
  if ( (_QWORD *)qword_4FD6AC8 == (_QWORD *)((char *)v3 + v4) )
  {
    v10 = *(_QWORD *)qword_4FD6AC8 + v4;
    *(_QWORD *)(a1 - 8) = v6;
    *(_QWORD *)(a1 - 16) = v10;
    qword_4FD6AC8 = a1 - 16;
    goto LABEL_17;
  }
  v7 = &qword_4FD6AC8;
  if ( v6 )
  {
    while ( 1 )
    {
      if ( v5 >= v6 )
      {
        if ( v5 == v6 )
        {
          v4 += *v5;
          v2[1] = v5[1];
        }
        v2 = (__int64 *)*v7;
        goto LABEL_14;
      }
      v7 = v2 + 1;
      if ( !v6[1] )
        break;
      v2 = v6;
      v6 = (__int64 *)v6[1];
    }
    v2 = (__int64 *)v2[1];
    v8 = *v2;
    if ( v3 != (__int64 *)((char *)v2 + *v2) )
      goto LABEL_15;
    goto LABEL_20;
  }
LABEL_14:
  v8 = *v2;
  if ( v3 == (__int64 *)((char *)v2 + *v2) )
  {
LABEL_20:
    *v2 = v4 + v8;
    goto LABEL_17;
  }
LABEL_15:
  v9 = v2[1];
  *(_QWORD *)(a1 - 16) = v4;
  *(_QWORD *)(a1 - 8) = v9;
  v2 = (__int64 *)*v7;
  *(_QWORD *)(*v7 + 8) = v3;
LABEL_17:
  if ( &_pthread_key_create )
  {
    LODWORD(v2) = pthread_mutex_unlock(&stru_4FD6AA0);
    if ( (_DWORD)v2 )
      JUMPOUT(0x42657E);
  }
  return (int)v2;
}
