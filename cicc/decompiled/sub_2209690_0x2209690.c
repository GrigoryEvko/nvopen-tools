// Function: sub_2209690
// Address: 0x2209690
//
void __fastcall sub_2209690(__int64 a1, volatile signed __int32 *a2, __int64 a3)
{
  _QWORD *v3; // rax
  _QWORD *v4; // rax
  volatile signed __int64 **v8; // rbx
  volatile signed __int64 *v9; // rdi
  __int64 v10; // rax
  volatile signed __int64 *v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rbp
  volatile signed __int32 **v15; // rax
  void (__fastcall *v16)(unsigned __int64); // rax
  __int64 v17; // rax
  volatile signed __int32 *v18; // rsi

  if ( !byte_4FD4F60 && (unsigned int)sub_2207590((__int64)&byte_4FD4F60) )
  {
    stru_4FD4F80.__list.__next = 0;
    *(_OWORD *)&stru_4FD4F80.__lock = 0;
    *((_OWORD *)&stru_4FD4F80.__align + 1) = 0;
    sub_2207640((__int64)&byte_4FD4F60);
  }
  if ( &_pthread_key_create && pthread_mutex_lock(&stru_4FD4F80) )
  {
    v3 = (_QWORD *)sub_2252770(8);
    *v3 = off_4A04690;
    sub_2253480(v3, &`typeinfo for'__gnu_cxx::__concurrence_lock_error, sub_2208D30);
  }
  v8 = (volatile signed __int64 **)&off_4A046E0;
  v9 = (volatile signed __int64 *)&unk_4FD69B8;
  if ( &unk_4FD69B8 )
  {
    while ( 1 )
    {
      v10 = sub_22091A0(v9);
      v11 = v8[1];
      if ( v10 == a3 )
      {
        v12 = sub_22091A0(v11);
        goto LABEL_10;
      }
      if ( sub_22091A0(v11) == a3 )
        break;
      v9 = v8[2];
      v8 += 2;
      if ( !v9 )
        goto LABEL_17;
    }
    v17 = sub_22091A0(*v8);
    v12 = a3;
    a3 = v17;
  }
  else
  {
LABEL_17:
    v12 = -1;
  }
LABEL_10:
  v13 = *(_QWORD *)(a1 + 24);
  v14 = 8 * a3;
  v15 = (volatile signed __int32 **)(v13 + v14);
  if ( *(_QWORD *)(v13 + v14) )
  {
    if ( a2 )
    {
      v16 = *(void (__fastcall **)(unsigned __int64))(*(_QWORD *)a2 + 8LL);
      if ( v16 == sub_2208CE0 )
      {
        nullsub_801();
        j___libc_free_0((unsigned __int64)a2);
      }
      else
      {
        ((void (__fastcall *)(volatile signed __int32 *, volatile signed __int32 *, void (__fastcall *)(unsigned __int64), __int64))v16)(
          a2,
          a2,
          sub_2208CE0,
          v12);
      }
    }
  }
  else
  {
    v18 = a2 + 2;
    if ( &_pthread_key_create )
    {
      _InterlockedAdd(v18, 1u);
      v13 = *(_QWORD *)(a1 + 24);
      v15 = (volatile signed __int32 **)(v13 + v14);
    }
    else
    {
      ++*((_DWORD *)a2 + 2);
    }
    *v15 = a2;
    if ( v12 != -1 )
    {
      if ( &_pthread_key_create )
      {
        _InterlockedAdd(v18, 1u);
        v13 = *(_QWORD *)(a1 + 24);
      }
      else
      {
        ++*((_DWORD *)a2 + 2);
      }
      *(_QWORD *)(v13 + 8 * v12) = a2;
    }
  }
  if ( &_pthread_key_create )
  {
    if ( pthread_mutex_unlock(&stru_4FD4F80) )
    {
      v4 = (_QWORD *)sub_2252770(8);
      *v4 = off_4A046B8;
      sub_2253480(v4, &`typeinfo for'__gnu_cxx::__concurrence_unlock_error, sub_2208D70);
    }
  }
}
