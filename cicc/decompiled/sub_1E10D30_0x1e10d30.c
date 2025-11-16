// Function: sub_1E10D30
// Address: 0x1e10d30
//
__int64 __fastcall sub_1E10D30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // r14
  unsigned __int64 v9; // rdi
  __int64 v10; // r14
  __int64 v11; // rbx
  unsigned __int64 v12; // r13
  __int64 v13; // rdi
  _QWORD *v15; // r15
  __int64 v16; // rbx
  __int64 v17; // rax
  _QWORD *v18; // rbx
  __int64 v19; // rdx
  __int64 v20; // rdi
  __int64 v21; // rax
  _QWORD *v22; // rbx
  _QWORD *v23; // r13
  void *v24; // [rsp+0h] [rbp-90h]
  __int64 v25; // [rsp+8h] [rbp-88h] BYREF
  __int64 v26; // [rsp+10h] [rbp-80h]
  __int64 v27; // [rsp+18h] [rbp-78h]
  __int64 v28; // [rsp+20h] [rbp-70h]
  void *v29; // [rsp+30h] [rbp-60h]
  _QWORD v30[2]; // [rsp+38h] [rbp-58h] BYREF
  __int64 v31; // [rsp+48h] [rbp-48h]
  __int64 v32; // [rsp+50h] [rbp-40h]

  if ( *(_BYTE *)(a1 + 216) )
  {
    v21 = *(unsigned int *)(a1 + 208);
    if ( (_DWORD)v21 )
    {
      v22 = *(_QWORD **)(a1 + 192);
      v23 = &v22[2 * v21];
      do
      {
        if ( *v22 != -8 && *v22 != -4 )
        {
          a2 = v22[1];
          if ( a2 )
            sub_161E7C0((__int64)(v22 + 1), a2);
        }
        v22 += 2;
      }
      while ( v23 != v22 );
    }
    j___libc_free_0(*(_QWORD *)(a1 + 192));
  }
  v7 = *(unsigned int *)(a1 + 176);
  if ( (_DWORD)v7 )
  {
    v15 = *(_QWORD **)(a1 + 160);
    v25 = 2;
    v16 = 6 * v7;
    v26 = 0;
    v17 = -8;
    v27 = -8;
    v18 = &v15[v16];
    v24 = &unk_49FB768;
    v28 = 0;
    v30[0] = 2;
    v30[1] = 0;
    v31 = -16;
    v29 = &unk_49FB768;
    v32 = 0;
    while ( 1 )
    {
      v19 = v15[3];
      if ( v19 != v17 )
      {
        v17 = v31;
        if ( v19 != v31 )
        {
          v20 = v15[5];
          v17 = v15[3];
          if ( v20 )
          {
            (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64, void *, __int64, __int64))(*(_QWORD *)v20 + 16LL))(
              v20,
              a2,
              v19,
              a4,
              a5,
              a6,
              v24,
              v25,
              v26);
            v17 = v15[3];
          }
        }
      }
      *v15 = &unk_49EE2B0;
      LOBYTE(a4) = v17 != -8;
      if ( ((v17 != 0) & (unsigned __int8)a4) != 0 && v17 != -16 )
        sub_1649B30(v15 + 1);
      v15 += 6;
      if ( v18 == v15 )
        break;
      v17 = v27;
    }
    v29 = &unk_49EE2B0;
    if ( v31 != -8 && v31 != 0 && v31 != -16 )
      sub_1649B30(v30);
    if ( v27 != -8 && v27 != 0 && v27 != -16 )
      sub_1649B30(&v25);
  }
  j___libc_free_0(*(_QWORD *)(a1 + 160));
  if ( *(_DWORD *)(a1 + 132) )
  {
    v8 = *(unsigned int *)(a1 + 128);
    v9 = *(_QWORD *)(a1 + 120);
    if ( (_DWORD)v8 )
    {
      v10 = 8 * v8;
      v11 = 0;
      do
      {
        v12 = *(_QWORD *)(v9 + v11);
        if ( v12 != -8 && v12 )
        {
          v13 = *(_QWORD *)(v12 + 8);
          if ( v13 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v13 + 16LL))(v13);
          _libc_free(v12);
          v9 = *(_QWORD *)(a1 + 120);
        }
        v11 += 8;
      }
      while ( v10 != v11 );
    }
  }
  else
  {
    v9 = *(_QWORD *)(a1 + 120);
  }
  _libc_free(v9);
  sub_1E09820(*(_QWORD **)(a1 + 88));
  nullsub_736(a1 + 56);
  nullsub_736(a1 + 40);
  nullsub_736(a1 + 24);
  nullsub_736(a1 + 8);
  return j_j___libc_free_0(a1, 232);
}
