// Function: sub_3210B10
// Address: 0x3210b10
//
__int64 __fastcall sub_3210B10(__int64 a1)
{
  unsigned __int64 *v2; // rbx
  unsigned __int64 *v3; // r12
  unsigned __int64 *v4; // rbx
  unsigned __int64 *v5; // r12
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  __int64 v12; // rax
  _QWORD *v13; // rbx
  _QWORD *v14; // r12
  unsigned __int64 *v15; // r13
  __int64 v16; // rax
  __int64 v17; // r15
  unsigned __int64 v18; // r12
  unsigned __int64 v19; // r13
  unsigned __int64 v20; // rbx
  unsigned __int64 v21; // rdi
  __int64 *v22; // r13
  __int64 *v23; // rbx
  __int64 i; // rax
  __int64 v25; // rdi
  unsigned int v26; // ecx
  __int64 v27; // rsi
  __int64 *v28; // rbx
  unsigned __int64 v29; // r12
  __int64 v30; // rsi
  __int64 v31; // rdi
  unsigned __int64 v32; // rdi
  bool v34; // cc
  unsigned __int64 v35; // rdi
  __int64 v36; // [rsp+0h] [rbp-40h]
  __int64 v37; // [rsp+8h] [rbp-38h]

  *(_QWORD *)a1 = off_4A354A0;
  sub_31F5340(*(_QWORD **)(a1 + 1408));
  v2 = *(unsigned __int64 **)(a1 + 1376);
  v3 = *(unsigned __int64 **)(a1 + 1368);
  if ( v2 != v3 )
  {
    do
    {
      if ( (unsigned __int64 *)*v3 != v3 + 2 )
        j_j___libc_free_0(*v3);
      v3 += 5;
    }
    while ( v2 != v3 );
    v3 = *(unsigned __int64 **)(a1 + 1368);
  }
  if ( v3 )
    j_j___libc_free_0((unsigned __int64)v3);
  v4 = *(unsigned __int64 **)(a1 + 1352);
  v5 = *(unsigned __int64 **)(a1 + 1344);
  if ( v4 != v5 )
  {
    do
    {
      if ( (unsigned __int64 *)*v5 != v5 + 2 )
        j_j___libc_free_0(*v5);
      v5 += 5;
    }
    while ( v4 != v5 );
    v5 = *(unsigned __int64 **)(a1 + 1344);
  }
  if ( v5 )
    j_j___libc_free_0((unsigned __int64)v5);
  v6 = *(_QWORD *)(a1 + 1280);
  if ( v6 != a1 + 1296 )
    _libc_free(v6);
  sub_C7D6A0(*(_QWORD *)(a1 + 1256), 16LL * *(unsigned int *)(a1 + 1272), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 1224), 24LL * *(unsigned int *)(a1 + 1240), 8);
  v7 = *(_QWORD *)(a1 + 1168);
  if ( v7 != a1 + 1184 )
    _libc_free(v7);
  sub_C7D6A0(*(_QWORD *)(a1 + 1144), 8LL * *(unsigned int *)(a1 + 1160), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 1112), 24LL * *(unsigned int *)(a1 + 1128), 8);
  sub_31F9F70(*(_QWORD *)(a1 + 1088), *(_QWORD *)(a1 + 1088) + 16LL * *(unsigned int *)(a1 + 1096));
  v8 = *(_QWORD *)(a1 + 1088);
  if ( v8 != a1 + 1104 )
    _libc_free(v8);
  sub_C7D6A0(*(_QWORD *)(a1 + 1064), 16LL * *(unsigned int *)(a1 + 1080), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 1024), 8LL * *(unsigned int *)(a1 + 1040), 8);
  v9 = *(_QWORD *)(a1 + 968);
  if ( v9 != a1 + 984 )
    _libc_free(v9);
  v10 = *(_QWORD *)(a1 + 936);
  if ( v10 != a1 + 952 )
    _libc_free(v10);
  v11 = *(_QWORD *)(a1 + 904);
  if ( v11 != a1 + 920 )
    _libc_free(v11);
  v12 = *(unsigned int *)(a1 + 896);
  if ( (_DWORD)v12 )
  {
    v13 = *(_QWORD **)(a1 + 880);
    v14 = &v13[2 * v12];
    do
    {
      if ( *v13 != -4096 && *v13 != -8192 )
      {
        v15 = (unsigned __int64 *)v13[1];
        if ( v15 )
        {
          if ( (unsigned __int64 *)*v15 != v15 + 2 )
            _libc_free(*v15);
          j_j___libc_free_0((unsigned __int64)v15);
        }
      }
      v13 += 2;
    }
    while ( v14 != v13 );
    LODWORD(v12) = *(_DWORD *)(a1 + 896);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 880), 16LL * (unsigned int)v12, 8);
  v16 = *(unsigned int *)(a1 + 864);
  if ( (_DWORD)v16 )
  {
    v17 = *(_QWORD *)(a1 + 848);
    v36 = v17 + 112 * v16;
    do
    {
      if ( *(_QWORD *)v17 != -8192 && *(_QWORD *)v17 != -4096 )
      {
        v37 = *(_QWORD *)(v17 + 8);
        v18 = v37 + 88LL * *(unsigned int *)(v17 + 16);
        if ( v37 != v18 )
        {
          do
          {
            v18 -= 88LL;
            if ( *(_BYTE *)(v18 + 80) )
            {
              v34 = *(_DWORD *)(v18 + 72) <= 0x40u;
              *(_BYTE *)(v18 + 80) = 0;
              if ( !v34 )
              {
                v35 = *(_QWORD *)(v18 + 64);
                if ( v35 )
                  j_j___libc_free_0_0(v35);
              }
            }
            v19 = *(_QWORD *)(v18 + 40);
            v20 = v19 + 40LL * *(unsigned int *)(v18 + 48);
            if ( v19 != v20 )
            {
              do
              {
                v20 -= 40LL;
                v21 = *(_QWORD *)(v20 + 8);
                if ( v21 != v20 + 24 )
                  _libc_free(v21);
              }
              while ( v19 != v20 );
              v19 = *(_QWORD *)(v18 + 40);
            }
            if ( v19 != v18 + 56 )
              _libc_free(v19);
            sub_C7D6A0(*(_QWORD *)(v18 + 16), 12LL * *(unsigned int *)(v18 + 32), 4);
          }
          while ( v37 != v18 );
          v18 = *(_QWORD *)(v17 + 8);
        }
        if ( v18 != v17 + 24 )
          _libc_free(v18);
      }
      v17 += 112;
    }
    while ( v36 != v17 );
    v16 = *(unsigned int *)(a1 + 864);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 848), 112 * v16, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 816), 16LL * *(unsigned int *)(a1 + 832), 8);
  sub_3707B10(a1 + 632);
  v22 = *(__int64 **)(a1 + 552);
  v23 = &v22[*(unsigned int *)(a1 + 560)];
  if ( v22 != v23 )
  {
    for ( i = *(_QWORD *)(a1 + 552); ; i = *(_QWORD *)(a1 + 552) )
    {
      v25 = *v22;
      v26 = (unsigned int)(((__int64)v22 - i) >> 3) >> 7;
      v27 = 4096LL << v26;
      if ( v26 >= 0x1E )
        v27 = 0x40000000000LL;
      ++v22;
      sub_C7D6A0(v25, v27, 16);
      if ( v23 == v22 )
        break;
    }
  }
  v28 = *(__int64 **)(a1 + 600);
  v29 = (unsigned __int64)&v28[2 * *(unsigned int *)(a1 + 608)];
  if ( v28 != (__int64 *)v29 )
  {
    do
    {
      v30 = v28[1];
      v31 = *v28;
      v28 += 2;
      sub_C7D6A0(v31, v30, 16);
    }
    while ( (__int64 *)v29 != v28 );
    v29 = *(_QWORD *)(a1 + 600);
  }
  if ( v29 != a1 + 616 )
    _libc_free(v29);
  v32 = *(_QWORD *)(a1 + 552);
  if ( v32 != a1 + 568 )
    _libc_free(v32);
  return sub_3212B10(a1);
}
