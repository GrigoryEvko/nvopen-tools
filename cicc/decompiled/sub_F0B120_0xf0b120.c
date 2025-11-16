// Function: sub_F0B120
// Address: 0xf0b120
//
unsigned __int8 *__fastcall sub_F0B120(__int64 a1, unsigned int **a2)
{
  __int64 v3; // r8
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 *v6; // rcx
  _BYTE *v7; // rbx
  _QWORD *v8; // rdx
  int v9; // esi
  __int64 v10; // rcx
  _QWORD *v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r9
  __int64 v14; // rax
  __int64 v15; // rbx
  unsigned __int8 *v16; // rax
  unsigned __int8 *v17; // r12
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rdx
  _BYTE *v25; // [rsp+10h] [rbp-B0h]
  unsigned int v26; // [rsp+18h] [rbp-A8h]
  __int64 v27; // [rsp+18h] [rbp-A8h]
  _QWORD *v28; // [rsp+18h] [rbp-A8h]
  __int64 v29; // [rsp+20h] [rbp-A0h]
  __int64 v30; // [rsp+20h] [rbp-A0h]
  int v31; // [rsp+20h] [rbp-A0h]
  __int64 v32; // [rsp+28h] [rbp-98h]
  const char *v33[4]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v34; // [rsp+50h] [rbp-70h]
  _BYTE **v35; // [rsp+60h] [rbp-60h] BYREF
  __int64 v36; // [rsp+68h] [rbp-58h]
  _BYTE v37[80]; // [rsp+70h] [rbp-50h] BYREF

  if ( !(unsigned __int8)sub_B4DD90(a1) )
    return 0;
  v4 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  v5 = *(_QWORD *)(a1 - 32 * v4);
  if ( *(_BYTE *)v5 != 86 )
    return 0;
  if ( (*(_BYTE *)(v5 + 7) & 0x40) != 0 )
  {
    v6 = *(__int64 **)(v5 - 8);
    v32 = *v6;
    if ( *v6 )
      goto LABEL_5;
    return 0;
  }
  v6 = (__int64 *)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF));
  v32 = *v6;
  if ( !*v6 )
    return 0;
LABEL_5:
  v7 = (_BYTE *)v6[4];
  if ( *v7 > 0x15u )
    return 0;
  v25 = (_BYTE *)v6[8];
  if ( *v25 > 0x15u )
    return 0;
  v36 = 0x400000000LL;
  v8 = v37;
  v9 = 0;
  v10 = 32 * (1 - v4);
  v35 = (_BYTE **)v37;
  v11 = (_QWORD *)(a1 + v10);
  v12 = -v10;
  v13 = v12 >> 5;
  if ( (unsigned __int64)v12 > 0x80 )
  {
    v28 = v11;
    v31 = v12 >> 5;
    sub_C8D5F0((__int64)&v35, v37, v12 >> 5, 8u, v3, v13);
    v9 = v36;
    v11 = v28;
    LODWORD(v13) = v31;
    v8 = &v35[(unsigned int)v36];
  }
  if ( (_QWORD *)a1 != v11 )
  {
    do
    {
      if ( v8 )
        *v8 = *v11;
      v11 += 4;
      ++v8;
    }
    while ( (_QWORD *)a1 != v11 );
    v9 = v36;
  }
  LODWORD(v36) = v9 + v13;
  v26 = sub_B4DE20(a1);
  v29 = *(_QWORD *)(a1 + 72);
  v34 = 257;
  v14 = sub_921130(a2, v29, (__int64)v7, v35, (unsigned int)v36, (__int64)v33, v26);
  v34 = 257;
  v15 = v14;
  v27 = sub_921130(a2, v29, (__int64)v25, v35, (unsigned int)v36, (__int64)v33, v26);
  v34 = 257;
  v16 = (unsigned __int8 *)sub_BD2C40(72, 3u);
  v17 = v16;
  if ( v16 )
  {
    v30 = (__int64)v16;
    sub_B44260((__int64)v16, *(_QWORD *)(v15 + 8), 57, 3u, 0, 0);
    if ( *((_QWORD *)v17 - 12) )
    {
      v18 = *((_QWORD *)v17 - 11);
      **((_QWORD **)v17 - 10) = v18;
      if ( v18 )
        *(_QWORD *)(v18 + 16) = *((_QWORD *)v17 - 10);
    }
    *((_QWORD *)v17 - 12) = v32;
    v19 = *(_QWORD *)(v32 + 16);
    *((_QWORD *)v17 - 11) = v19;
    if ( v19 )
      *(_QWORD *)(v19 + 16) = v17 - 88;
    *((_QWORD *)v17 - 10) = v32 + 16;
    *(_QWORD *)(v32 + 16) = v17 - 96;
    if ( *((_QWORD *)v17 - 8) )
    {
      v20 = *((_QWORD *)v17 - 7);
      **((_QWORD **)v17 - 6) = v20;
      if ( v20 )
        *(_QWORD *)(v20 + 16) = *((_QWORD *)v17 - 6);
    }
    *((_QWORD *)v17 - 8) = v15;
    v21 = *(_QWORD *)(v15 + 16);
    *((_QWORD *)v17 - 7) = v21;
    if ( v21 )
      *(_QWORD *)(v21 + 16) = v17 - 56;
    *((_QWORD *)v17 - 6) = v15 + 16;
    *(_QWORD *)(v15 + 16) = v17 - 64;
    if ( *((_QWORD *)v17 - 4) )
    {
      v22 = *((_QWORD *)v17 - 3);
      **((_QWORD **)v17 - 2) = v22;
      if ( v22 )
        *(_QWORD *)(v22 + 16) = *((_QWORD *)v17 - 2);
    }
    *((_QWORD *)v17 - 4) = v27;
    if ( v27 )
    {
      v23 = *(_QWORD *)(v27 + 16);
      *((_QWORD *)v17 - 3) = v23;
      if ( v23 )
        *(_QWORD *)(v23 + 16) = v17 - 24;
      *((_QWORD *)v17 - 2) = v27 + 16;
      *(_QWORD *)(v27 + 16) = v17 - 32;
    }
    sub_BD6B50(v17, v33);
  }
  else
  {
    v30 = 0;
  }
  sub_B47C00(v30, v5, 0, 0);
  if ( v35 != (_BYTE **)v37 )
    _libc_free(v35, v5);
  return v17;
}
