// Function: sub_16F12D0
// Address: 0x16f12d0
//
void *__fastcall sub_16F12D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  void *v10; // r12
  __int64 v11; // r14
  _QWORD *v12; // rsi
  _BYTE *v14; // rsi
  void *v15; // rax
  void *v16; // [rsp+0h] [rbp-30h] BYREF
  void *v17; // [rsp+8h] [rbp-28h] BYREF

  if ( !qword_4FA1800 )
    sub_16C1EA0((__int64)&qword_4FA1800, (__int64 (*)(void))sub_16F0F80, (__int64)sub_16F10B0, a4, a5, a6);
  v6 = qword_4FA1800;
  v10 = sub_16F10F0(a1, a2);
  if ( v10 == &unk_4FA17DA )
    return v10;
  if ( qword_4FA17E0 )
  {
    v11 = qword_4FA17E0;
    if ( !(unsigned __int8)sub_16D5D40() )
    {
LABEL_6:
      ++*(_DWORD *)(v11 + 8);
      goto LABEL_7;
    }
  }
  else
  {
    sub_16C1EA0((__int64)&qword_4FA17E0, sub_160CFB0, (__int64)sub_160D0B0, v7, v8, v9);
    v11 = qword_4FA17E0;
    if ( !(unsigned __int8)sub_16D5D40() )
      goto LABEL_6;
  }
  sub_16C30C0((pthread_mutex_t **)v11);
LABEL_7:
  v16 = v10;
  if ( a1 )
  {
    v17 = v10;
    v12 = *(_QWORD **)(v6 + 8);
    if ( v12 == sub_16F0FC0(*(_QWORD **)v6, (__int64)v12, (__int64 *)&v17) )
    {
      v14 = *(_BYTE **)(v6 + 8);
      if ( v14 == *(_BYTE **)(v6 + 16) )
      {
        sub_16F1140(v6, v14, &v16);
      }
      else
      {
        if ( v14 )
        {
          *(_QWORD *)v14 = v16;
          v14 = *(_BYTE **)(v6 + 8);
        }
        *(_QWORD *)(v6 + 8) = v14 + 8;
      }
    }
    else
    {
      nullsub_632();
    }
  }
  else
  {
    if ( *(_QWORD *)(v6 + 24) )
    {
      nullsub_632();
      v15 = v16;
      if ( *(void **)(v6 + 24) == v16 )
        goto LABEL_10;
    }
    else
    {
      v15 = v10;
    }
    *(_QWORD *)(v6 + 24) = v15;
  }
LABEL_10:
  if ( (unsigned __int8)sub_16D5D40() )
  {
    sub_16C30E0((pthread_mutex_t **)v11);
    return v10;
  }
  --*(_DWORD *)(v11 + 8);
  return v10;
}
