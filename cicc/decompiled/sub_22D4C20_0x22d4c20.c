// Function: sub_22D4C20
// Address: 0x22d4c20
//
__int64 __fastcall sub_22D4C20(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r13
  unsigned __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // r13
  unsigned __int64 v14; // rdi
  __int64 v15; // rsi
  __int64 v16; // rdi
  _QWORD *v18; // rbx
  _QWORD *v19; // r15
  __int64 v20; // rax
  _QWORD v21[2]; // [rsp+8h] [rbp-88h] BYREF
  __int64 v22; // [rsp+18h] [rbp-78h]
  __int64 v23; // [rsp+20h] [rbp-70h]
  void *v24; // [rsp+30h] [rbp-60h]
  _QWORD v25[2]; // [rsp+38h] [rbp-58h] BYREF
  __int64 v26; // [rsp+48h] [rbp-48h]
  __int64 v27; // [rsp+50h] [rbp-40h]

  v2 = sub_22077B0(0x90u);
  if ( v2 )
  {
    *(_DWORD *)v2 = 1;
    *(_QWORD *)(v2 + 8) = 0;
    *(_QWORD *)(v2 + 16) = 0;
    *(_QWORD *)(v2 + 24) = 0;
    *(_DWORD *)(v2 + 32) = 0;
    *(_QWORD *)(v2 + 40) = 0;
    *(_QWORD *)(v2 + 48) = 0;
    *(_QWORD *)(v2 + 56) = 0;
    *(_DWORD *)(v2 + 64) = 0;
    *(_QWORD *)(v2 + 72) = 0;
    *(_QWORD *)(v2 + 80) = 0;
    *(_QWORD *)(v2 + 88) = 0;
    *(_DWORD *)(v2 + 96) = 0;
    *(_QWORD *)(v2 + 104) = 0;
    *(_QWORD *)(v2 + 112) = 0;
    *(_QWORD *)(v2 + 120) = 0;
    *(_DWORD *)(v2 + 128) = 0;
    *(_QWORD *)(v2 + 136) = a2;
  }
  v3 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = v2;
  if ( v3 )
  {
    v4 = *(unsigned int *)(v3 + 128);
    if ( (_DWORD)v4 )
    {
      v18 = *(_QWORD **)(v3 + 112);
      v21[0] = 2;
      v21[1] = 0;
      v22 = -4096;
      v19 = &v18[5 * v4];
      v23 = 0;
      v25[0] = 2;
      v25[1] = 0;
      v26 = -8192;
      v24 = &unk_4A09ED8;
      v27 = 0;
      do
      {
        v20 = v18[3];
        *v18 = &unk_49DB368;
        if ( v20 != 0 && v20 != -4096 && v20 != -8192 )
          sub_BD60C0(v18 + 1);
        v18 += 5;
      }
      while ( v19 != v18 );
      v24 = &unk_49DB368;
      if ( v26 != 0 && v26 != -4096 && v26 != -8192 )
        sub_BD60C0(v25);
      if ( v22 != 0 && v22 != -4096 && v22 != -8192 )
        sub_BD60C0(v21);
      v4 = *(unsigned int *)(v3 + 128);
    }
    sub_C7D6A0(*(_QWORD *)(v3 + 112), 40 * v4, 8);
    v5 = *(unsigned int *)(v3 + 96);
    if ( (_DWORD)v5 )
    {
      v6 = *(_QWORD *)(v3 + 80);
      v7 = v6 + 88 * v5;
      do
      {
        while ( *(_DWORD *)v6 > 0xFFFFFFFD )
        {
          v6 += 88;
          if ( v7 == v6 )
            goto LABEL_12;
        }
        v8 = *(_QWORD *)(v6 + 40);
        if ( v8 != v6 + 56 )
          _libc_free(v8);
        v9 = *(unsigned int *)(v6 + 32);
        v10 = *(_QWORD *)(v6 + 16);
        v6 += 88;
        sub_C7D6A0(v10, 8 * v9, 8);
      }
      while ( v7 != v6 );
LABEL_12:
      v5 = *(unsigned int *)(v3 + 96);
    }
    sub_C7D6A0(*(_QWORD *)(v3 + 80), 88 * v5, 8);
    v11 = *(unsigned int *)(v3 + 64);
    if ( (_DWORD)v11 )
    {
      v12 = *(_QWORD *)(v3 + 48);
      v13 = v12 + 88 * v11;
      do
      {
        while ( *(_DWORD *)v12 > 0xFFFFFFFD )
        {
          v12 += 88;
          if ( v13 == v12 )
            goto LABEL_20;
        }
        v14 = *(_QWORD *)(v12 + 40);
        if ( v14 != v12 + 56 )
          _libc_free(v14);
        v15 = *(unsigned int *)(v12 + 32);
        v16 = *(_QWORD *)(v12 + 16);
        v12 += 88;
        sub_C7D6A0(v16, 8 * v15, 8);
      }
      while ( v13 != v12 );
LABEL_20:
      v11 = *(unsigned int *)(v3 + 64);
    }
    sub_C7D6A0(*(_QWORD *)(v3 + 48), 88 * v11, 8);
    sub_C7D6A0(*(_QWORD *)(v3 + 16), 16LL * *(unsigned int *)(v3 + 32), 8);
    j_j___libc_free_0(v3);
  }
  return 0;
}
