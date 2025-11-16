// Function: sub_2E81360
// Address: 0x2e81360
//
void __fastcall sub_2E81360(unsigned __int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // r14
  unsigned __int64 v9; // rdi
  __int64 v10; // r14
  __int64 v11; // rbx
  _QWORD *v12; // r12
  __int64 v13; // rdi
  __int64 v14; // r15
  __int64 v15; // rbx
  unsigned __int64 v16; // r12
  __int64 v17; // rdi
  _QWORD *v18; // r15
  __int64 v19; // r12
  __int64 v20; // rax
  _QWORD *v21; // r12
  __int64 v22; // rdx
  __int64 v23; // rdi
  __int64 v24; // rax
  _QWORD *v25; // rbx
  _QWORD *v26; // r12
  __int64 v27; // rsi
  void *v28; // [rsp+0h] [rbp-90h]
  __int64 v29; // [rsp+8h] [rbp-88h] BYREF
  __int64 v30; // [rsp+10h] [rbp-80h]
  __int64 v31; // [rsp+18h] [rbp-78h]
  __int64 v32; // [rsp+20h] [rbp-70h]
  void *v33; // [rsp+30h] [rbp-60h]
  _QWORD v34[2]; // [rsp+38h] [rbp-58h] BYREF
  __int64 v35; // [rsp+48h] [rbp-48h]
  __int64 v36; // [rsp+50h] [rbp-40h]

  if ( *(_BYTE *)(a1 + 224) )
  {
    v24 = *(unsigned int *)(a1 + 216);
    *(_BYTE *)(a1 + 224) = 0;
    if ( (_DWORD)v24 )
    {
      v25 = *(_QWORD **)(a1 + 200);
      v26 = &v25[2 * v24];
      do
      {
        if ( *v25 != -8192 && *v25 != -4096 )
        {
          v27 = v25[1];
          if ( v27 )
            sub_B91220((__int64)(v25 + 1), v27);
        }
        v25 += 2;
      }
      while ( v26 != v25 );
      LODWORD(v24) = *(_DWORD *)(a1 + 216);
    }
    a2 = 16LL * (unsigned int)v24;
    sub_C7D6A0(*(_QWORD *)(a1 + 200), a2, 8);
  }
  v7 = *(unsigned int *)(a1 + 184);
  if ( (_DWORD)v7 )
  {
    v18 = *(_QWORD **)(a1 + 168);
    v29 = 2;
    v19 = 6 * v7;
    v30 = 0;
    v20 = -4096;
    v31 = -4096;
    v21 = &v18[v19];
    v28 = &unk_4A28E90;
    v32 = 0;
    v34[0] = 2;
    v34[1] = 0;
    v35 = -8192;
    v33 = &unk_4A28E90;
    v36 = 0;
    while ( 1 )
    {
      v22 = v18[3];
      if ( v22 != v20 )
      {
        v20 = v35;
        if ( v22 != v35 )
        {
          v23 = v18[5];
          v20 = v18[3];
          if ( v23 )
          {
            (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64, void *, __int64, __int64))(*(_QWORD *)v23 + 16LL))(
              v23,
              a2,
              v22,
              a4,
              a5,
              a6,
              v28,
              v29,
              v30);
            v20 = v18[3];
          }
        }
      }
      *v18 = &unk_49DB368;
      LOBYTE(a4) = v20 != -4096;
      if ( ((v20 != 0) & (unsigned __int8)a4) != 0 && v20 != -8192 )
        sub_BD60C0(v18 + 1);
      v18 += 6;
      if ( v21 == v18 )
        break;
      v20 = v31;
    }
    v33 = &unk_49DB368;
    if ( v35 != -4096 && v35 != 0 && v35 != -8192 )
      sub_BD60C0(v34);
    if ( v31 != -4096 && v31 != 0 && v31 != -8192 )
      sub_BD60C0(&v29);
    v7 = *(unsigned int *)(a1 + 184);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 168), 48 * v7, 8);
  if ( *(_DWORD *)(a1 + 148) )
  {
    v8 = *(unsigned int *)(a1 + 144);
    v9 = *(_QWORD *)(a1 + 136);
    if ( (_DWORD)v8 )
    {
      v10 = 8 * v8;
      v11 = 0;
      do
      {
        v12 = *(_QWORD **)(v9 + v11);
        if ( v12 && v12 != (_QWORD *)-8LL )
        {
          v13 = v12[1];
          v14 = *v12 + 17LL;
          if ( v13 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v13 + 16LL))(v13);
          sub_C7D6A0((__int64)v12, v14, 8);
          v9 = *(_QWORD *)(a1 + 136);
        }
        v11 += 8;
      }
      while ( v10 != v11 );
    }
  }
  else
  {
    v9 = *(_QWORD *)(a1 + 136);
  }
  _libc_free(v9);
  v15 = *(_QWORD *)(a1 + 72);
  v16 = v15 + 8LL * *(unsigned int *)(a1 + 80);
  if ( v15 != v16 )
  {
    do
    {
      v17 = *(_QWORD *)(v16 - 8);
      v16 -= 8LL;
      if ( v17 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v17 + 16LL))(v17);
    }
    while ( v15 != v16 );
    v16 = *(_QWORD *)(a1 + 72);
  }
  if ( v16 != a1 + 88 )
    _libc_free(v16);
  nullsub_1642(a1 + 56);
  nullsub_1642(a1 + 40);
  nullsub_1642(a1 + 24);
  nullsub_1642(a1 + 8);
  j_j___libc_free_0(a1);
}
