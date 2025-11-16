// Function: sub_2397090
// Address: 0x2397090
//
__int64 __fastcall sub_2397090(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r13
  unsigned __int64 v6; // rdi
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r13
  unsigned __int64 v12; // rdi
  __int64 v13; // rsi
  __int64 v14; // rdi
  _QWORD *v16; // rbx
  _QWORD *v17; // r15
  __int64 v18; // rax
  _QWORD v19[5]; // [rsp+8h] [rbp-88h] BYREF
  void *v20; // [rsp+30h] [rbp-60h]
  _QWORD v21[11]; // [rsp+38h] [rbp-58h] BYREF

  v2 = *(unsigned int *)(a1 + 128);
  if ( (_DWORD)v2 )
  {
    v16 = *(_QWORD **)(a1 + 112);
    v19[0] = 2;
    v19[1] = 0;
    v19[2] = -4096;
    v17 = &v16[5 * v2];
    v19[3] = 0;
    v21[0] = 2;
    v21[1] = 0;
    v21[2] = -8192;
    v20 = &unk_4A09ED8;
    v21[3] = 0;
    do
    {
      v18 = v16[3];
      *v16 = &unk_49DB368;
      if ( v18 != -4096 && v18 != 0 && v18 != -8192 )
        sub_BD60C0(v16 + 1);
      v16 += 5;
    }
    while ( v17 != v16 );
    v20 = &unk_49DB368;
    sub_D68D70(v21);
    sub_D68D70(v19);
    v2 = *(unsigned int *)(a1 + 128);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 112), 40 * v2, 8);
  v3 = *(unsigned int *)(a1 + 96);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD *)(a1 + 80);
    v5 = v4 + 88 * v3;
    do
    {
      while ( *(_DWORD *)v4 > 0xFFFFFFFD )
      {
        v4 += 88;
        if ( v5 == v4 )
          goto LABEL_9;
      }
      v6 = *(_QWORD *)(v4 + 40);
      if ( v6 != v4 + 56 )
        _libc_free(v6);
      v7 = *(unsigned int *)(v4 + 32);
      v8 = *(_QWORD *)(v4 + 16);
      v4 += 88;
      sub_C7D6A0(v8, 8 * v7, 8);
    }
    while ( v5 != v4 );
LABEL_9:
    v3 = *(unsigned int *)(a1 + 96);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 80), 88 * v3, 8);
  v9 = *(unsigned int *)(a1 + 64);
  if ( (_DWORD)v9 )
  {
    v10 = *(_QWORD *)(a1 + 48);
    v11 = v10 + 88 * v9;
    do
    {
      while ( *(_DWORD *)v10 > 0xFFFFFFFD )
      {
        v10 += 88;
        if ( v11 == v10 )
          goto LABEL_17;
      }
      v12 = *(_QWORD *)(v10 + 40);
      if ( v12 != v10 + 56 )
        _libc_free(v12);
      v13 = *(unsigned int *)(v10 + 32);
      v14 = *(_QWORD *)(v10 + 16);
      v10 += 88;
      sub_C7D6A0(v14, 8 * v13, 8);
    }
    while ( v11 != v10 );
LABEL_17:
    v9 = *(unsigned int *)(a1 + 64);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 48), 88 * v9, 8);
  return sub_C7D6A0(*(_QWORD *)(a1 + 16), 16LL * *(unsigned int *)(a1 + 32), 8);
}
