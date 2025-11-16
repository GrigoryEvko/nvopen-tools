// Function: sub_2673100
// Address: 0x2673100
//
void __fastcall sub_2673100(__int64 a1, __int64 a2)
{
  __int64 v2; // r9
  __int64 v3; // r12
  unsigned int v4; // eax
  __int64 v5; // rdx
  __int64 v6; // r8
  __int64 v7; // r13
  __int64 *v8; // r15
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rcx
  unsigned __int64 v12; // rdx
  unsigned __int8 *v13; // rax
  __int64 *v14; // rdi
  __int64 v15; // rsi
  _QWORD *v16; // rax
  __int64 v17; // [rsp+0h] [rbp-80h]
  __int64 v18; // [rsp+8h] [rbp-78h]
  __int64 v19; // [rsp+10h] [rbp-70h]
  __int64 v20; // [rsp+10h] [rbp-70h]
  unsigned __int8 *v21; // [rsp+10h] [rbp-70h]
  __int64 *v22; // [rsp+18h] [rbp-68h]
  __int64 *v23; // [rsp+20h] [rbp-60h] BYREF
  __int64 v24; // [rsp+28h] [rbp-58h]
  _BYTE v25[80]; // [rsp+30h] [rbp-50h] BYREF

  v2 = a1;
  v3 = *(_QWORD *)(a2 + 208);
  v18 = a1 + 184;
  ++*(_QWORD *)(a1 + 184);
  if ( *(_BYTE *)(a1 + 212) )
    goto LABEL_6;
  v4 = 4 * (*(_DWORD *)(a1 + 204) - *(_DWORD *)(a1 + 208));
  v5 = *(unsigned int *)(a1 + 200);
  if ( v4 < 0x20 )
    v4 = 32;
  if ( (unsigned int)v5 <= v4 )
  {
    memset(*(void **)(a1 + 192), -1, 8 * v5);
    v2 = a1;
LABEL_6:
    *(_QWORD *)(v2 + 204) = 0;
    goto LABEL_7;
  }
  sub_C8C990(v18, a2);
  v2 = a1;
LABEL_7:
  v6 = *(_QWORD *)(v2 + 136);
  v7 = 0x8000000000041LL;
  v8 = (__int64 *)v6;
  v22 = (__int64 *)(v6 + 8LL * *(unsigned int *)(v2 + 144));
  if ( (__int64 *)v6 == v22 )
    return;
  do
  {
    v9 = *v8;
    v23 = (__int64 *)v25;
    v24 = 0x400000000LL;
    v10 = *(_QWORD *)(v9 + 16);
    if ( !v10 )
      goto LABEL_26;
    v11 = 0;
    do
    {
      while ( 1 )
      {
        v13 = *(unsigned __int8 **)(v10 + 24);
        v12 = *v13;
        if ( (unsigned __int8)v12 <= 0x1Cu )
          goto LABEL_13;
        v12 = (unsigned int)(v12 - 34);
        if ( (unsigned __int8)v12 > 0x33u || !_bittest64(&v7, v12) )
          goto LABEL_13;
        v12 = *((_QWORD *)v13 - 4);
        if ( !v12 )
          goto LABEL_12;
        if ( *(_BYTE *)v12 )
          break;
        if ( *(_QWORD *)(v12 + 24) != *((_QWORD *)v13 + 10) )
          v12 = 0;
LABEL_12:
        if ( *(_QWORD *)(v3 + 32592) == v12 )
          goto LABEL_20;
LABEL_13:
        v10 = *(_QWORD *)(v10 + 8);
        if ( !v10 )
          goto LABEL_23;
      }
      v12 = 0;
      if ( *(_QWORD *)(v3 + 32592) )
        goto LABEL_13;
LABEL_20:
      if ( v11 + 1 > (unsigned __int64)HIDWORD(v24) )
      {
        v17 = v2;
        v21 = *(unsigned __int8 **)(v10 + 24);
        sub_C8D5F0((__int64)&v23, v25, v11 + 1, 8u, v6, v2);
        v11 = (unsigned int)v24;
        v2 = v17;
        v13 = v21;
      }
      v12 = (unsigned __int64)v23;
      v23[v11] = (__int64)v13;
      v11 = (unsigned int)(v24 + 1);
      LODWORD(v24) = v24 + 1;
      v10 = *(_QWORD *)(v10 + 8);
    }
    while ( v10 );
LABEL_23:
    v14 = v23;
    if ( (_DWORD)v11 != 1 )
      goto LABEL_24;
    v15 = *v23;
    if ( !*(_BYTE *)(v2 + 212) )
      goto LABEL_34;
    v16 = *(_QWORD **)(v2 + 192);
    v11 = *(unsigned int *)(v2 + 204);
    v12 = (unsigned __int64)&v16[v11];
    if ( v16 == (_QWORD *)v12 )
    {
LABEL_32:
      if ( (unsigned int)v11 < *(_DWORD *)(v2 + 200) )
      {
        *(_DWORD *)(v2 + 204) = v11 + 1;
        *(_QWORD *)v12 = v15;
        v14 = v23;
        ++*(_QWORD *)(v2 + 184);
        goto LABEL_24;
      }
LABEL_34:
      v20 = v2;
      sub_C8CC70(v18, v15, v12, v11, v6, v2);
      v14 = v23;
      v2 = v20;
      goto LABEL_24;
    }
    while ( v15 != *v16 )
    {
      if ( (_QWORD *)v12 == ++v16 )
        goto LABEL_32;
    }
LABEL_24:
    if ( v14 != (__int64 *)v25 )
    {
      v19 = v2;
      _libc_free((unsigned __int64)v14);
      v2 = v19;
    }
LABEL_26:
    ++v8;
  }
  while ( v22 != v8 );
}
