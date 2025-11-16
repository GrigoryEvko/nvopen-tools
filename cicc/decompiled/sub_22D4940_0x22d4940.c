// Function: sub_22D4940
// Address: 0x22d4940
//
void __fastcall sub_22D4940(_QWORD *a1)
{
  unsigned __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r14
  unsigned __int64 v7; // rdi
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v12; // r14
  unsigned __int64 v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // rdi
  _QWORD *v16; // rbx
  _QWORD *v17; // r14
  __int64 v18; // rax
  _QWORD v19[2]; // [rsp+8h] [rbp-88h] BYREF
  __int64 v20; // [rsp+18h] [rbp-78h]
  __int64 v21; // [rsp+20h] [rbp-70h]
  void *v22; // [rsp+30h] [rbp-60h]
  _QWORD v23[2]; // [rsp+38h] [rbp-58h] BYREF
  __int64 v24; // [rsp+48h] [rbp-48h]
  __int64 v25; // [rsp+50h] [rbp-40h]

  v2 = a1[22];
  *a1 = &unk_4A09F00;
  if ( v2 )
  {
    v3 = *(unsigned int *)(v2 + 128);
    if ( (_DWORD)v3 )
    {
      v16 = *(_QWORD **)(v2 + 112);
      v19[0] = 2;
      v19[1] = 0;
      v17 = &v16[5 * v3];
      v20 = -4096;
      v21 = 0;
      v23[0] = 2;
      v23[1] = 0;
      v24 = -8192;
      v22 = &unk_4A09ED8;
      v25 = 0;
      do
      {
        v18 = v16[3];
        *v16 = &unk_49DB368;
        if ( v18 != 0 && v18 != -4096 && v18 != -8192 )
          sub_BD60C0(v16 + 1);
        v16 += 5;
      }
      while ( v17 != v16 );
      v22 = &unk_49DB368;
      if ( v24 != 0 && v24 != -4096 && v24 != -8192 )
        sub_BD60C0(v23);
      if ( v20 != 0 && v20 != -4096 && v20 != -8192 )
        sub_BD60C0(v19);
      v3 = *(unsigned int *)(v2 + 128);
    }
    sub_C7D6A0(*(_QWORD *)(v2 + 112), 40 * v3, 8);
    v4 = *(unsigned int *)(v2 + 96);
    if ( (_DWORD)v4 )
    {
      v5 = *(_QWORD *)(v2 + 80);
      v6 = v5 + 88 * v4;
      do
      {
        while ( *(_DWORD *)v5 > 0xFFFFFFFD )
        {
          v5 += 88;
          if ( v6 == v5 )
            goto LABEL_10;
        }
        v7 = *(_QWORD *)(v5 + 40);
        if ( v7 != v5 + 56 )
          _libc_free(v7);
        v8 = *(unsigned int *)(v5 + 32);
        v9 = *(_QWORD *)(v5 + 16);
        v5 += 88;
        sub_C7D6A0(v9, 8 * v8, 8);
      }
      while ( v6 != v5 );
LABEL_10:
      v4 = *(unsigned int *)(v2 + 96);
    }
    sub_C7D6A0(*(_QWORD *)(v2 + 80), 88 * v4, 8);
    v10 = *(unsigned int *)(v2 + 64);
    if ( (_DWORD)v10 )
    {
      v11 = *(_QWORD *)(v2 + 48);
      v12 = v11 + 88 * v10;
      do
      {
        while ( *(_DWORD *)v11 > 0xFFFFFFFD )
        {
          v11 += 88;
          if ( v12 == v11 )
            goto LABEL_18;
        }
        v13 = *(_QWORD *)(v11 + 40);
        if ( v13 != v11 + 56 )
          _libc_free(v13);
        v14 = *(unsigned int *)(v11 + 32);
        v15 = *(_QWORD *)(v11 + 16);
        v11 += 88;
        sub_C7D6A0(v15, 8 * v14, 8);
      }
      while ( v12 != v11 );
LABEL_18:
      v10 = *(unsigned int *)(v2 + 64);
    }
    sub_C7D6A0(*(_QWORD *)(v2 + 48), 88 * v10, 8);
    sub_C7D6A0(*(_QWORD *)(v2 + 16), 16LL * *(unsigned int *)(v2 + 32), 8);
    j_j___libc_free_0(v2);
  }
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
