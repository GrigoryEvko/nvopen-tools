// Function: sub_14CAD00
// Address: 0x14cad00
//
__int64 __fastcall sub_14CAD00(__int64 a1)
{
  __int64 v2; // rdx
  __int64 v3; // r13
  bool v4; // al
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r15
  void *v8; // rsi
  __int64 v9; // r13
  __int64 result; // rax
  __int64 v11; // r14
  unsigned __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // rax
  bool v15; // zf
  __int64 v16; // rdx
  unsigned int v17; // esi
  __int64 v18; // rdi
  int v19; // r10d
  __int64 v20; // [rsp+8h] [rbp-58h] BYREF
  __int64 v21; // [rsp+10h] [rbp-50h]
  __int64 v22; // [rsp+18h] [rbp-48h]
  __int64 v23; // [rsp+20h] [rbp-40h]

  v2 = *(_QWORD *)(a1 + 24);
  v20 = 2;
  v21 = 0;
  v3 = *(_QWORD *)(a1 + 32);
  v22 = v2;
  v4 = v2 != -16 && v2 != 0 && v2 != -8;
  if ( v4 )
  {
    sub_164C220(&v20);
    v2 = v22;
    v4 = v22 != -16 && v22 != 0 && v22 != -8;
  }
  v23 = 0;
  v5 = *(unsigned int *)(v3 + 176);
  v6 = *(_QWORD *)(v3 + 160);
  if ( !(_DWORD)v5 )
  {
LABEL_4:
    v7 = v6 + 88 * v5;
    goto LABEL_5;
  }
  v17 = (v5 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
  v7 = v6 + 88LL * v17;
  v18 = *(_QWORD *)(v7 + 24);
  if ( v2 != v18 )
  {
    v19 = 1;
    while ( v18 != -8 )
    {
      v17 = (v5 - 1) & (v19 + v17);
      v7 = v6 + 88LL * v17;
      v18 = *(_QWORD *)(v7 + 24);
      if ( v2 == v18 )
        goto LABEL_5;
      ++v19;
    }
    goto LABEL_4;
  }
LABEL_5:
  v8 = &unk_49EE2A0;
  if ( v4 )
    sub_1649B30(&v20);
  v9 = *(_QWORD *)(a1 + 32);
  result = *(_QWORD *)(v9 + 160) + 88LL * *(unsigned int *)(v9 + 176);
  if ( v7 != result )
  {
    v11 = *(_QWORD *)(v7 + 40);
    v12 = v11 + 32LL * *(unsigned int *)(v7 + 48);
    if ( v11 != v12 )
    {
      do
      {
        v13 = *(_QWORD *)(v12 - 16);
        v12 -= 32LL;
        if ( v13 != -8 && v13 != 0 && v13 != -16 )
          sub_1649B30(v12);
      }
      while ( v11 != v12 );
      v12 = *(_QWORD *)(v7 + 40);
    }
    if ( v12 != v7 + 56 )
      _libc_free(v12);
    v20 = 2;
    v21 = 0;
    v22 = -16;
    v23 = 0;
    result = *(_QWORD *)(v7 + 24);
    if ( result == -16 )
    {
      *(_QWORD *)(v7 + 32) = 0;
    }
    else
    {
      if ( result == -8 || !result )
      {
        *(_QWORD *)(v7 + 24) = -16;
LABEL_22:
        v16 = v22;
        LOBYTE(result) = v22 != -8;
        v15 = v22 == 0;
        *(_QWORD *)(v7 + 32) = v23;
        LOBYTE(v8) = !v15;
        LOBYTE(v16) = v16 != -16;
        result = (unsigned int)v16 & (unsigned int)v8 & (unsigned int)result;
        if ( (_BYTE)result )
          result = sub_1649B30(&v20);
        goto LABEL_24;
      }
      sub_1649B30(v7 + 8);
      v14 = v22;
      v15 = v22 == -8;
      *(_QWORD *)(v7 + 24) = v22;
      if ( v14 != 0 && !v15 && v14 != -16 )
      {
        LODWORD(v8) = v20 & 0xFFFFFFF8;
        result = sub_1649AC0(v7 + 8, v20 & 0xFFFFFFFFFFFFFFF8LL);
        goto LABEL_22;
      }
      result = v23;
      *(_QWORD *)(v7 + 32) = v23;
    }
LABEL_24:
    --*(_DWORD *)(v9 + 168);
    ++*(_DWORD *)(v9 + 172);
  }
  return result;
}
