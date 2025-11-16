// Function: sub_2EBA1B0
// Address: 0x2eba1b0
//
void __fastcall sub_2EBA1B0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 *v4; // rdi
  __int64 v5; // rbx
  unsigned __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rbx
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rax
  int v12; // r15d
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rdx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rdx
  unsigned __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rbx
  __int64 *v34; // r14
  __int64 *v35; // rax
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  _BYTE *v39; // rbx
  unsigned __int64 v40; // r12
  unsigned __int64 v41; // rdi
  char *v43[2]; // [rsp+20h] [rbp-1090h] BYREF
  char v44; // [rsp+30h] [rbp-1080h] BYREF
  unsigned __int64 v45[2]; // [rsp+50h] [rbp-1060h] BYREF
  _QWORD v46[64]; // [rsp+60h] [rbp-1050h] BYREF
  _BYTE *v47; // [rsp+260h] [rbp-E50h]
  __int64 v48; // [rsp+268h] [rbp-E48h]
  _BYTE v49[3584]; // [rsp+270h] [rbp-E40h] BYREF
  __int64 v50; // [rsp+1070h] [rbp-40h]

  v2 = 0;
  v4 = (__int64 *)(a1 + 48);
  v5 = v4[10];
  sub_2E6DCE0(v4);
  *(_DWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_BYTE *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 140) = 0;
  *(_QWORD *)(a1 + 128) = v5;
  if ( a2 )
  {
    v2 = *(_QWORD *)(a2 + 16);
    if ( v2 )
    {
      v9 = *(_QWORD *)(a2 + 8);
      if ( v2 != v9 )
        sub_2EB4EE0(*(_QWORD *)(a2 + 8), *(_QWORD *)(a2 + 16));
      if ( v2 + 304 != v9 + 304 )
        sub_2EB4EE0(v9 + 304, v2 + 304);
      *(_BYTE *)(v9 + 608) = *(_BYTE *)(v2 + 608);
      if ( v9 + 616 == v2 + 616 )
      {
        v2 = a2;
      }
      else
      {
        v10 = *(unsigned int *)(v2 + 624);
        v11 = *(unsigned int *)(v9 + 624);
        v12 = *(_DWORD *)(v2 + 624);
        if ( v10 <= v11 )
        {
          if ( *(_DWORD *)(v2 + 624) )
            memmove(*(void **)(v9 + 616), *(const void **)(v2 + 616), 16 * v10);
        }
        else
        {
          v6 = *(unsigned int *)(v9 + 628);
          if ( v10 > v6 )
          {
            v13 = 0;
            *(_DWORD *)(v9 + 624) = 0;
            sub_C8D5F0(v9 + 616, (const void *)(v9 + 632), v10, 0x10u, v7, v8);
            v10 = *(unsigned int *)(v2 + 624);
          }
          else
          {
            v13 = 16 * v11;
            if ( *(_DWORD *)(v9 + 624) )
            {
              memmove(*(void **)(v9 + 616), *(const void **)(v2 + 616), 16 * v11);
              v10 = *(unsigned int *)(v2 + 624);
            }
          }
          v14 = *(_QWORD *)(v2 + 616);
          v15 = 16 * v10;
          if ( v14 + v13 != v15 + v14 )
            memcpy((void *)(v13 + *(_QWORD *)(v9 + 616)), (const void *)(v14 + v13), v15 - v13);
        }
        *(_DWORD *)(v9 + 624) = v12;
        v2 = a2;
      }
    }
  }
  v45[0] = (unsigned __int64)v46;
  v45[1] = 0x4000000001LL;
  v47 = v49;
  v48 = 0x4000000000LL;
  v46[0] = 0;
  v50 = v2;
  sub_2EB9A60(v43, a1, v2, v6, v7, v8);
  sub_2EB3190(a1, v43, v16, v17, v18, v19);
  if ( v43[0] != &v44 )
    _libc_free((unsigned __int64)v43[0]);
  v24 = sub_2EB5B40((__int64)v45, 0, v20, v21, v22, v23);
  v25 = 0;
  *(_QWORD *)(v24 + 8) = 0x100000001LL;
  *(_DWORD *)v24 = 1;
  sub_2E6D5A0((__int64)v45, 0, v26, 0x100000001LL, v27, v28);
  v33 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v33 )
  {
    v34 = *(__int64 **)a1;
    v29 = 1;
    do
    {
      v25 = *v34++;
      v29 = (unsigned int)sub_2EB8890(
                            (__int64)v45,
                            v25,
                            v29,
                            (unsigned __int8 (__fastcall *)(__int64, _QWORD))sub_2EB2D60,
                            1,
                            0);
    }
    while ( (__int64 *)v33 != v34 );
  }
  sub_2EB5CF0((__int64)v45, v25, v29, v30, v31, v32);
  if ( a2 )
    *(_BYTE *)a2 = 1;
  if ( *(_DWORD *)(a1 + 8) )
  {
    v35 = (__int64 *)sub_2EB4C20(a1, 0, 0);
    *(_QWORD *)(a1 + 120) = v35;
    sub_2EB61B0(v45, a1, *v35, v36, v37, v38);
    v39 = v47;
    v40 = (unsigned __int64)&v47[56 * (unsigned int)v48];
    if ( v47 != (_BYTE *)v40 )
    {
      do
      {
        v40 -= 56LL;
        v41 = *(_QWORD *)(v40 + 24);
        if ( v41 != v40 + 40 )
          _libc_free(v41);
      }
      while ( v39 != (_BYTE *)v40 );
      v40 = (unsigned __int64)v47;
    }
    if ( (_BYTE *)v40 != v49 )
      _libc_free(v40);
    if ( (_QWORD *)v45[0] != v46 )
      _libc_free(v45[0]);
  }
  else
  {
    sub_2EB40F0((__int64)v45);
  }
}
