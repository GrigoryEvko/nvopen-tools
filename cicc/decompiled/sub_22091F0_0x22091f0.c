// Function: sub_22091F0
// Address: 0x22091f0
//
void __fastcall sub_22091F0(_QWORD *a1, volatile signed __int64 *a2, volatile signed __int32 *a3)
{
  unsigned __int64 v5; // rbp
  unsigned __int64 v6; // r14
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // r15
  __int64 v9; // rax
  unsigned __int64 v10; // rcx
  __int64 v11; // r13
  __int64 v12; // rsi
  __int64 i; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // rdi
  __int64 j; // rax
  volatile signed __int32 **v19; // r14
  volatile signed __int32 *v20; // r15
  volatile signed __int64 *v21; // rdi
  _QWORD *v22; // r15
  __int64 v23; // rbp
  volatile signed __int32 **v24; // rbp
  __int64 v25; // rax
  volatile signed __int32 *v26; // r15
  signed __int32 v27; // eax
  int v28; // eax
  bool v29; // zf
  __int64 v30; // rax
  unsigned __int64 v31; // rbp
  volatile signed __int32 *v32; // r12
  signed __int32 v33; // eax
  void (__fastcall *v34)(unsigned __int64); // rax
  void (__fastcall *v35)(unsigned __int64); // rax
  __int64 v36; // rbp
  __int64 v37; // rax
  signed __int32 v38; // eax
  volatile signed __int32 *v39; // rdi
  void (__fastcall *v40)(unsigned __int64); // rax
  unsigned __int64 v41; // [rsp+0h] [rbp-50h]
  volatile signed __int32 *v42; // [rsp+0h] [rbp-50h]
  unsigned __int64 v43; // [rsp+8h] [rbp-48h]
  __int64 v44; // [rsp+10h] [rbp-40h]

  if ( !a3 )
    return;
  v5 = sub_22091A0(a2);
  if ( a1[2] - 1LL < v5 )
  {
    v6 = v5 + 4;
    v41 = a1[1];
    v7 = -1;
    if ( v5 + 4 <= 0xFFFFFFFFFFFFFFFLL )
      v7 = 8 * (v5 + 4);
    v8 = v7;
    v9 = sub_2207820(v7);
    v10 = a1[2];
    v11 = v9;
    if ( v10 )
    {
      v12 = a1[1];
      for ( i = 0; i != v10; ++i )
        *(_QWORD *)(v11 + 8 * i) = *(_QWORD *)(v12 + 8 * i);
    }
    if ( v10 < v6 )
      memset((void *)(v11 + 8 * v10), 0, 8 * (v6 - v10));
    v43 = a1[3];
    v14 = sub_2207820(v8);
    v15 = a1[2];
    v16 = v14;
    if ( v15 )
    {
      v17 = a1[3];
      for ( j = 0; j != v15; ++j )
        *(_QWORD *)(v16 + 8 * j) = *(_QWORD *)(v17 + 8 * j);
    }
    if ( v6 > v15 )
    {
      v44 = v16;
      memset((void *)(v16 + 8 * v15), 0, 8 * (v6 - v15));
      v16 = v44;
    }
    a1[2] = v6;
    a1[1] = v11;
    a1[3] = v16;
    if ( v41 )
      j_j___libc_free_0_0(v41);
    if ( v43 )
      j_j___libc_free_0_0(v43);
  }
  if ( &_pthread_key_create )
    _InterlockedAdd(a3 + 2, 1u);
  else
    ++*((_DWORD *)a3 + 2);
  v19 = (volatile signed __int32 **)(a1[1] + 8 * v5);
  v20 = *v19;
  if ( *v19 )
  {
    v21 = (volatile signed __int64 *)&unk_4FD69B8;
    if ( &unk_4FD69B8 )
    {
      v22 = &off_4A046E0;
      while ( 1 )
      {
        if ( sub_22091A0(v21) == v5 )
        {
          v23 = a1[1];
          v24 = (volatile signed __int32 **)(v23 + 8 * sub_22091A0((volatile signed __int64 *)v22[1]));
          if ( *v24 )
          {
            v25 = sub_2222A70(a3, v22[1]);
            v42 = (volatile signed __int32 *)v25;
            if ( &_pthread_key_create )
              _InterlockedAdd((volatile signed __int32 *)(v25 + 8), 1u);
            else
              ++*(_DWORD *)(v25 + 8);
            v26 = *v24;
            if ( &_pthread_key_create )
            {
              v27 = _InterlockedExchangeAdd(v26 + 2, 0xFFFFFFFF);
            }
            else
            {
              v27 = *((_DWORD *)v26 + 2);
              *((_DWORD *)v26 + 2) = v27 - 1;
            }
            if ( v27 == 1 )
            {
              v39 = v26;
              v40 = *(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v26 + 8LL);
              if ( v40 != sub_2208CE0 )
                goto LABEL_70;
LABEL_62:
              nullsub_801();
              j___libc_free_0((unsigned __int64)v26);
            }
            goto LABEL_33;
          }
          goto LABEL_49;
        }
        if ( sub_22091A0((volatile signed __int64 *)v22[1]) == v5 )
          break;
        v21 = (volatile signed __int64 *)v22[2];
        v22 += 2;
        if ( !v21 )
          goto LABEL_49;
      }
      v36 = a1[1];
      v24 = (volatile signed __int32 **)(v36 + 8 * sub_22091A0((volatile signed __int64 *)*v22));
      if ( *v24 )
      {
        v37 = sub_2214870(a3, *v22);
        v42 = (volatile signed __int32 *)v37;
        if ( &_pthread_key_create )
          _InterlockedAdd((volatile signed __int32 *)(v37 + 8), 1u);
        else
          ++*(_DWORD *)(v37 + 8);
        v26 = *v24;
        if ( &_pthread_key_create )
        {
          v38 = _InterlockedExchangeAdd(v26 + 2, 0xFFFFFFFF);
        }
        else
        {
          v38 = *((_DWORD *)v26 + 2);
          *((_DWORD *)v26 + 2) = v38 - 1;
        }
        if ( v38 == 1 )
        {
          v39 = v26;
          v40 = *(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v26 + 8LL);
          if ( v40 == sub_2208CE0 )
            goto LABEL_62;
LABEL_70:
          v40((unsigned __int64)v39);
        }
LABEL_33:
        *v24 = v42;
        v20 = *v19;
        goto LABEL_34;
      }
LABEL_49:
      v20 = *v19;
      if ( &_pthread_key_create )
      {
LABEL_50:
        if ( _InterlockedExchangeAdd(v20 + 2, 0xFFFFFFFF) != 1 )
          goto LABEL_36;
        goto LABEL_51;
      }
    }
    else
    {
LABEL_34:
      if ( &_pthread_key_create )
        goto LABEL_50;
    }
    v28 = *((_DWORD *)v20 + 2);
    *((_DWORD *)v20 + 2) = v28 - 1;
    if ( v28 != 1 )
      goto LABEL_36;
LABEL_51:
    v35 = *(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v20 + 8LL);
    if ( v35 == sub_2208CE0 )
    {
      nullsub_801();
      j___libc_free_0((unsigned __int64)v20);
    }
    else
    {
      v35((unsigned __int64)v20);
    }
  }
LABEL_36:
  v29 = a1[2] == 0;
  *v19 = a3;
  if ( !v29 )
  {
    v30 = a1[3];
    v31 = 0;
    do
    {
      v32 = *(volatile signed __int32 **)(v30 + 8 * v31);
      if ( v32 )
      {
        if ( &_pthread_key_create )
        {
          v33 = _InterlockedExchangeAdd(v32 + 2, 0xFFFFFFFF);
        }
        else
        {
          v33 = *((_DWORD *)v32 + 2);
          *((_DWORD *)v32 + 2) = v33 - 1;
        }
        if ( v33 == 1 )
        {
          v34 = *(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v32 + 8LL);
          if ( v34 == sub_2208CE0 )
          {
            nullsub_801();
            j___libc_free_0((unsigned __int64)v32);
          }
          else
          {
            v34((unsigned __int64)v32);
          }
        }
        v30 = a1[3];
        *(_QWORD *)(v30 + 8 * v31) = 0;
      }
      ++v31;
    }
    while ( a1[2] > v31 );
  }
}
