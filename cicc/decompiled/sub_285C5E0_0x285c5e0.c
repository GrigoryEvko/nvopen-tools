// Function: sub_285C5E0
// Address: 0x285c5e0
//
__int64 __fastcall sub_285C5E0(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  _QWORD *v4; // rbx
  _QWORD *v5; // r12
  __int64 v6; // rax
  unsigned __int64 *v7; // rbx
  unsigned __int64 *v8; // r12
  unsigned __int64 v9; // rdi
  __int64 v10; // rax
  _QWORD *v11; // rbx
  _QWORD *v12; // r12
  unsigned __int64 v13; // r14
  __int64 v14; // r14
  unsigned __int64 v15; // r12
  unsigned __int64 v16; // r15
  unsigned __int64 v17; // rbx
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // r15
  unsigned __int64 v20; // rbx
  __int64 v21; // rdx
  unsigned __int64 *v22; // r15
  unsigned __int64 *v23; // rbx
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi

  sub_C7D6A0(*(_QWORD *)(a1 + 37504), 16LL * *(unsigned int *)(a1 + 37520), 8);
  v2 = *(_QWORD *)(a1 + 37096);
  if ( v2 != a1 + 37112 )
    _libc_free(v2);
  v3 = *(_QWORD *)(a1 + 37048);
  if ( v3 != a1 + 37064 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 37024), 8LL * *(unsigned int *)(a1 + 37040), 8);
  v4 = *(_QWORD **)(a1 + 36952);
  v5 = &v4[3 * *(unsigned int *)(a1 + 36960)];
  if ( v4 != v5 )
  {
    do
    {
      v6 = *(v5 - 1);
      v5 -= 3;
      if ( v6 != 0 && v6 != -4096 && v6 != -8192 )
        sub_BD60C0(v5);
    }
    while ( v4 != v5 );
    v5 = *(_QWORD **)(a1 + 36952);
  }
  if ( v5 != (_QWORD *)(a1 + 36968) )
    _libc_free((unsigned __int64)v5);
  if ( !*(_BYTE *)(a1 + 36884) )
    _libc_free(*(_QWORD *)(a1 + 36864));
  v7 = *(unsigned __int64 **)(a1 + 36456);
  v8 = &v7[6 * *(unsigned int *)(a1 + 36464)];
  if ( v7 != v8 )
  {
    do
    {
      v8 -= 6;
      if ( (unsigned __int64 *)*v8 != v8 + 2 )
        _libc_free(*v8);
    }
    while ( v7 != v8 );
    v8 = *(unsigned __int64 **)(a1 + 36456);
  }
  if ( v8 != (unsigned __int64 *)(a1 + 36472) )
    _libc_free((unsigned __int64)v8);
  v9 = *(_QWORD *)(a1 + 36312);
  if ( v9 != a1 + 36328 )
    _libc_free(v9);
  v10 = *(unsigned int *)(a1 + 36304);
  if ( (_DWORD)v10 )
  {
    v11 = *(_QWORD **)(a1 + 36288);
    v12 = &v11[2 * v10];
    do
    {
      if ( *v11 != -8192 && *v11 != -4096 )
      {
        v13 = v11[1];
        if ( (v13 & 1) == 0 )
        {
          if ( v13 )
          {
            if ( *(_QWORD *)v13 != v13 + 16 )
              _libc_free(*(_QWORD *)v13);
            j_j___libc_free_0(v13);
          }
        }
      }
      v11 += 2;
    }
    while ( v12 != v11 );
    LODWORD(v10) = *(_DWORD *)(a1 + 36304);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 36288), 16LL * (unsigned int)v10, 8);
  v14 = *(_QWORD *)(a1 + 1320);
  v15 = v14 + 2184LL * *(unsigned int *)(a1 + 1328);
  if ( v14 != v15 )
  {
    do
    {
      v15 -= 2184LL;
      if ( !*(_BYTE *)(v15 + 2148) )
        _libc_free(*(_QWORD *)(v15 + 2128));
      v16 = *(_QWORD *)(v15 + 760);
      v17 = v16 + 112LL * *(unsigned int *)(v15 + 768);
      if ( v16 != v17 )
      {
        do
        {
          v17 -= 112LL;
          v18 = *(_QWORD *)(v17 + 40);
          if ( v18 != v17 + 56 )
            _libc_free(v18);
        }
        while ( v16 != v17 );
        v16 = *(_QWORD *)(v15 + 760);
      }
      if ( v16 != v15 + 776 )
        _libc_free(v16);
      v19 = *(_QWORD *)(v15 + 56);
      v20 = v19 + 80LL * *(unsigned int *)(v15 + 64);
      if ( v19 != v20 )
      {
        do
        {
          while ( 1 )
          {
            v20 -= 80LL;
            if ( !*(_BYTE *)(v20 + 44) )
              break;
            if ( v19 == v20 )
              goto LABEL_50;
          }
          _libc_free(*(_QWORD *)(v20 + 24));
        }
        while ( v19 != v20 );
LABEL_50:
        v19 = *(_QWORD *)(v15 + 56);
      }
      if ( v19 != v15 + 72 )
        _libc_free(v19);
      v21 = *(unsigned int *)(v15 + 24);
      if ( (_DWORD)v21 )
      {
        v22 = *(unsigned __int64 **)(v15 + 8);
        v23 = &v22[6 * v21];
        do
        {
          if ( (unsigned __int64 *)*v22 != v22 + 2 )
            _libc_free(*v22);
          v22 += 6;
        }
        while ( v23 != v22 );
      }
      sub_C7D6A0(*(_QWORD *)(v15 + 8), 48LL * *(unsigned int *)(v15 + 24), 8);
    }
    while ( v14 != v15 );
    v15 = *(_QWORD *)(a1 + 1320);
  }
  if ( v15 != a1 + 1336 )
    _libc_free(v15);
  v24 = *(_QWORD *)(a1 + 1272);
  if ( v24 != a1 + 1288 )
    _libc_free(v24);
  sub_C7D6A0(*(_QWORD *)(a1 + 1248), 8LL * *(unsigned int *)(a1 + 1264), 8);
  v25 = *(_QWORD *)(a1 + 1096);
  if ( v25 != a1 + 1112 )
    _libc_free(v25);
  sub_28520F0(*(_QWORD *)(a1 + 1064));
  v26 = *(_QWORD *)(a1 + 968);
  if ( v26 != a1 + 984 )
    _libc_free(v26);
  return sub_27C20B0(a1 + 80);
}
