// Function: sub_1D5E530
// Address: 0x1d5e530
//
void *__fastcall sub_1D5E530(__int64 a1)
{
  __int64 v2; // rax
  _QWORD *v3; // rbx
  _QWORD *v4; // r13
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  __int64 v7; // r13
  _QWORD *v8; // rbx
  _QWORD *v9; // r13
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  __int64 v13; // r13
  __int64 v14; // r13
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  __int64 v17; // r13
  _QWORD *v19; // rbx
  _QWORD *i; // r15
  __int64 v21; // rax
  _QWORD *v22; // rbx
  _QWORD *v23; // r13
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rax
  _QWORD *v28; // rbx
  _QWORD *v29; // r13
  __int64 v30; // rsi
  __int64 v31; // [rsp+18h] [rbp-88h] BYREF
  __int64 v32; // [rsp+20h] [rbp-80h]
  __int64 v33; // [rsp+28h] [rbp-78h]
  __int64 v34; // [rsp+30h] [rbp-70h]
  void *v35; // [rsp+40h] [rbp-60h]
  __int64 v36; // [rsp+48h] [rbp-58h] BYREF
  __int64 v37; // [rsp+50h] [rbp-50h]
  __int64 v38; // [rsp+58h] [rbp-48h]
  __int64 v39; // [rsp+60h] [rbp-40h]

  *(_QWORD *)a1 = off_49F9D90;
  v2 = *(unsigned int *)(a1 + 888);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD **)(a1 + 872);
    v4 = &v3[19 * v2];
    do
    {
      if ( *v3 != -16 && *v3 != -8 )
      {
        v5 = v3[1];
        if ( (_QWORD *)v5 != v3 + 3 )
          _libc_free(v5);
      }
      v3 += 19;
    }
    while ( v4 != v3 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 872));
  j___libc_free_0(*(_QWORD *)(a1 + 840));
  sub_1D5ACA0(*(_QWORD *)(a1 + 800));
  v6 = *(_QWORD *)(a1 + 752);
  if ( v6 != a1 + 768 )
    _libc_free(v6);
  v7 = *(unsigned int *)(a1 + 744);
  if ( (_DWORD)v7 )
  {
    v8 = *(_QWORD **)(a1 + 728);
    v9 = &v8[67 * v7];
    do
    {
      if ( *v8 != -16 && *v8 != -8 )
      {
        v10 = v8[1];
        if ( (_QWORD *)v10 != v8 + 3 )
          _libc_free(v10);
      }
      v8 += 67;
    }
    while ( v9 != v8 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 728));
  j___libc_free_0(*(_QWORD *)(a1 + 696));
  v11 = *(_QWORD *)(a1 + 536);
  if ( v11 != *(_QWORD *)(a1 + 528) )
    _libc_free(v11);
  j___libc_free_0(*(_QWORD *)(a1 + 496));
  v12 = *(_QWORD *)(a1 + 336);
  if ( v12 != *(_QWORD *)(a1 + 328) )
    _libc_free(v12);
  if ( *(_BYTE *)(a1 + 304) )
  {
    v27 = *(unsigned int *)(a1 + 296);
    if ( (_DWORD)v27 )
    {
      v28 = *(_QWORD **)(a1 + 280);
      v29 = &v28[2 * v27];
      do
      {
        if ( *v28 != -4 && *v28 != -8 )
        {
          v30 = v28[1];
          if ( v30 )
            sub_161E7C0((__int64)(v28 + 1), v30);
        }
        v28 += 2;
      }
      while ( v29 != v28 );
    }
    j___libc_free_0(*(_QWORD *)(a1 + 280));
  }
  v13 = *(unsigned int *)(a1 + 264);
  if ( (_DWORD)v13 )
  {
    v22 = *(_QWORD **)(a1 + 248);
    v31 = 2;
    v32 = 0;
    v33 = -8;
    v23 = &v22[8 * v13];
    v35 = &unk_49F9E38;
    v24 = -8;
    v34 = 0;
    v36 = 2;
    v37 = 0;
    v38 = -16;
    v39 = 0;
    while ( 1 )
    {
      v25 = v22[3];
      if ( v25 != v24 )
      {
        v24 = v38;
        if ( v25 != v38 )
        {
          v26 = v22[7];
          if ( v26 != -8 && v26 != 0 && v26 != -16 )
          {
            sub_1649B30(v22 + 5);
            v25 = v22[3];
          }
          v24 = v25;
        }
      }
      *v22 = &unk_49EE2B0;
      if ( v24 != -8 && v24 != 0 && v24 != -16 )
        sub_1649B30(v22 + 1);
      v22 += 8;
      if ( v23 == v22 )
        break;
      v24 = v33;
    }
    v35 = &unk_49EE2B0;
    sub_1455FA0((__int64)&v36);
    sub_1455FA0((__int64)&v31);
  }
  j___libc_free_0(*(_QWORD *)(a1 + 248));
  v14 = *(_QWORD *)(a1 + 224);
  if ( v14 )
  {
    v15 = *(_QWORD *)(v14 + 256);
    if ( v15 != *(_QWORD *)(v14 + 248) )
      _libc_free(v15);
    v16 = *(_QWORD *)(v14 + 88);
    if ( v16 != *(_QWORD *)(v14 + 80) )
      _libc_free(v16);
    j___libc_free_0(*(_QWORD *)(v14 + 40));
    if ( *(_DWORD *)(v14 + 24) )
    {
      v32 = 0;
      v33 = -8;
      v34 = 0;
      v31 = 2;
      v36 = 2;
      v37 = 0;
      v38 = -16;
      v35 = &unk_49E8A80;
      v39 = 0;
      v19 = *(_QWORD **)(v14 + 8);
      for ( i = &v19[5 * *(unsigned int *)(v14 + 24)]; i != v19; v19 += 5 )
      {
        v21 = v19[3];
        *v19 = &unk_49EE2B0;
        if ( v21 != -8 && v21 != 0 && v21 != -16 )
          sub_1649B30(v19 + 1);
      }
      v35 = &unk_49EE2B0;
      sub_1455FA0((__int64)&v36);
      sub_1455FA0((__int64)&v31);
    }
    j___libc_free_0(*(_QWORD *)(v14 + 8));
    j_j___libc_free_0(v14, 408);
  }
  v17 = *(_QWORD *)(a1 + 216);
  if ( v17 )
  {
    sub_1368A00(*(__int64 **)(a1 + 216));
    j_j___libc_free_0(v17, 8);
  }
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0((_QWORD *)a1);
}
