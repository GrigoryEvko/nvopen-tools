// Function: sub_23CE6A0
// Address: 0x23ce6a0
//
__int64 __fastcall sub_23CE6A0(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rdi
  unsigned __int64 *v4; // r13
  unsigned __int64 *v5; // r12
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  volatile signed __int32 *v13; // r12
  signed __int32 v14; // eax
  _QWORD *v15; // r12
  __int64 (__fastcall *v16)(_QWORD *); // rax
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // r14
  __int64 (__fastcall *v23)(__int64); // rax
  unsigned __int64 *v24; // r13
  unsigned __int64 *v25; // r12
  __int64 v26; // rdi
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdi
  signed __int32 v31; // eax

  *(_QWORD *)a1 = &unk_4A16308;
  v3 = *(_QWORD *)(a1 + 1232);
  if ( v3 != a1 + 1248 )
  {
    a2 = *(_QWORD *)(a1 + 1248) + 1LL;
    j_j___libc_free_0(v3);
  }
  v4 = *(unsigned __int64 **)(a1 + 1208);
  v5 = *(unsigned __int64 **)(a1 + 1200);
  if ( v4 != v5 )
  {
    do
    {
      if ( (unsigned __int64 *)*v5 != v5 + 2 )
      {
        a2 = v5[2] + 1;
        j_j___libc_free_0(*v5);
      }
      v5 += 4;
    }
    while ( v4 != v5 );
    v5 = *(unsigned __int64 **)(a1 + 1200);
  }
  if ( v5 )
  {
    a2 = *(_QWORD *)(a1 + 1216) - (_QWORD)v5;
    j_j___libc_free_0((unsigned __int64)v5);
  }
  v6 = *(_QWORD *)(a1 + 1168);
  if ( v6 != a1 + 1184 )
  {
    a2 = *(_QWORD *)(a1 + 1184) + 1LL;
    j_j___libc_free_0(v6);
  }
  v7 = *(_QWORD *)(a1 + 1136);
  if ( v7 != a1 + 1152 )
  {
    a2 = *(_QWORD *)(a1 + 1152) + 1LL;
    j_j___libc_free_0(v7);
  }
  v8 = *(_QWORD *)(a1 + 1104);
  if ( v8 != a1 + 1120 )
  {
    a2 = *(_QWORD *)(a1 + 1120) + 1LL;
    j_j___libc_free_0(v8);
  }
  v9 = *(_QWORD *)(a1 + 1072);
  if ( v9 != a1 + 1088 )
  {
    a2 = *(_QWORD *)(a1 + 1088) + 1LL;
    j_j___libc_free_0(v9);
  }
  v10 = *(_QWORD *)(a1 + 1040);
  if ( v10 != a1 + 1056 )
  {
    a2 = *(_QWORD *)(a1 + 1056) + 1LL;
    j_j___libc_free_0(v10);
  }
  v11 = *(_QWORD *)(a1 + 1008);
  if ( v11 != a1 + 1024 )
  {
    a2 = *(_QWORD *)(a1 + 1024) + 1LL;
    j_j___libc_free_0(v11);
  }
  v12 = *(_QWORD *)(a1 + 912);
  if ( v12 != a1 + 928 )
  {
    a2 = *(_QWORD *)(a1 + 928) + 1LL;
    j_j___libc_free_0(v12);
  }
  v13 = *(volatile signed __int32 **)(a1 + 896);
  if ( v13 )
  {
    if ( &_pthread_key_create )
    {
      v14 = _InterlockedExchangeAdd(v13 + 2, 0xFFFFFFFF);
    }
    else
    {
      v14 = *((_DWORD *)v13 + 2);
      *((_DWORD *)v13 + 2) = v14 - 1;
    }
    if ( v14 == 1 )
    {
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v13 + 16LL))(v13);
      if ( &_pthread_key_create )
      {
        v31 = _InterlockedExchangeAdd(v13 + 3, 0xFFFFFFFF);
      }
      else
      {
        v31 = *((_DWORD *)v13 + 3);
        *((_DWORD *)v13 + 3) = v31 - 1;
      }
      if ( v31 == 1 )
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v13 + 24LL))(v13);
    }
  }
  if ( *(_BYTE *)(a1 + 848) )
  {
    *(_BYTE *)(a1 + 848) = 0;
    sub_23C66F0((unsigned __int64 *)(a1 + 696));
  }
  v15 = *(_QWORD **)(a1 + 680);
  if ( v15 )
  {
    v16 = *(__int64 (__fastcall **)(_QWORD *))(*v15 + 8LL);
    if ( v16 == sub_C12070 )
    {
      v17 = v15[34];
      *v15 = &unk_49E41D0;
      if ( (_QWORD *)v17 != v15 + 36 )
        j_j___libc_free_0(v17);
      v18 = v15[12];
      if ( (_QWORD *)v18 != v15 + 14 )
        j_j___libc_free_0(v18);
      v19 = v15[8];
      if ( (_QWORD *)v19 != v15 + 10 )
        j_j___libc_free_0(v19);
      v20 = v15[1];
      if ( (_QWORD *)v20 != v15 + 3 )
        j_j___libc_free_0(v20);
      a2 = 304;
      j_j___libc_free_0((unsigned __int64)v15);
    }
    else
    {
      v16(*(_QWORD **)(a1 + 680));
    }
  }
  v21 = *(_QWORD *)(a1 + 672);
  if ( v21 )
  {
    a2 = 48;
    j_j___libc_free_0(v21);
  }
  v22 = *(_QWORD *)(a1 + 664);
  if ( v22 )
  {
    v23 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v22 + 8LL);
    if ( v23 == sub_C11FA0 )
    {
      v24 = *(unsigned __int64 **)(v22 + 232);
      v25 = *(unsigned __int64 **)(v22 + 224);
      *(_QWORD *)v22 = &unk_49E3560;
      if ( v24 != v25 )
      {
        do
        {
          if ( *v25 )
            j_j___libc_free_0(*v25);
          v25 += 3;
        }
        while ( v24 != v25 );
        v25 = *(unsigned __int64 **)(v22 + 224);
      }
      if ( v25 )
        j_j___libc_free_0((unsigned __int64)v25);
      sub_C7D6A0(*(_QWORD *)(v22 + 200), 8LL * *(unsigned int *)(v22 + 216), 4);
      sub_C7D6A0(*(_QWORD *)(v22 + 168), 8LL * *(unsigned int *)(v22 + 184), 4);
      a2 = 248;
      j_j___libc_free_0(v22);
    }
    else
    {
      v23(*(_QWORD *)(a1 + 664));
    }
  }
  v26 = *(_QWORD *)(a1 + 656);
  if ( v26 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v26 + 8LL))(v26);
  v27 = *(_QWORD *)(a1 + 600);
  if ( v27 != a1 + 616 )
  {
    a2 = *(_QWORD *)(a1 + 616) + 1LL;
    j_j___libc_free_0(v27);
  }
  v28 = *(_QWORD *)(a1 + 568);
  if ( v28 != a1 + 584 )
  {
    a2 = *(_QWORD *)(a1 + 584) + 1LL;
    j_j___libc_free_0(v28);
  }
  v29 = *(_QWORD *)(a1 + 512);
  if ( v29 != a1 + 528 )
  {
    a2 = *(_QWORD *)(a1 + 528) + 1LL;
    j_j___libc_free_0(v29);
  }
  return sub_AE4030((_QWORD *)(a1 + 16), a2);
}
