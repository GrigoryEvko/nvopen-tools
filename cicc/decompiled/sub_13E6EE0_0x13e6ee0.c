// Function: sub_13E6EE0
// Address: 0x13e6ee0
//
__int64 __fastcall sub_13E6EE0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // r13
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  __int64 v15; // rax
  _QWORD *v16; // rbx
  _QWORD *v17; // r14
  __int64 v18; // rax
  _QWORD v20[2]; // [rsp+8h] [rbp-88h] BYREF
  __int64 v21; // [rsp+18h] [rbp-78h]
  __int64 v22; // [rsp+20h] [rbp-70h]
  void *v23; // [rsp+30h] [rbp-60h]
  _QWORD v24[2]; // [rsp+38h] [rbp-58h] BYREF
  __int64 v25; // [rsp+48h] [rbp-48h]
  __int64 v26; // [rsp+50h] [rbp-40h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_33:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F9920C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_33;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F9920C);
  v6 = *(__int64 **)(a1 + 8);
  v7 = v5 + 160;
  v8 = *v6;
  v9 = v6[1];
  if ( v8 == v9 )
LABEL_34:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4F9B6E8 )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_34;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4F9B6E8)
      + 360;
  v11 = sub_22077B0(440);
  if ( v11 )
  {
    *(_QWORD *)v11 = 0;
    *(_QWORD *)(v11 + 80) = v11 + 112;
    *(_QWORD *)(v11 + 88) = v11 + 112;
    *(_QWORD *)(v11 + 8) = 0;
    *(_QWORD *)(v11 + 16) = 0;
    *(_DWORD *)(v11 + 24) = 0;
    *(_QWORD *)(v11 + 32) = 0;
    *(_QWORD *)(v11 + 40) = 0;
    *(_QWORD *)(v11 + 48) = 0;
    *(_DWORD *)(v11 + 56) = 0;
    *(_QWORD *)(v11 + 72) = 0;
    *(_QWORD *)(v11 + 96) = 16;
    *(_DWORD *)(v11 + 104) = 0;
    *(_QWORD *)(v11 + 240) = 0;
    *(_QWORD *)(v11 + 248) = v11 + 280;
    *(_QWORD *)(v11 + 256) = v11 + 280;
    *(_QWORD *)(v11 + 264) = 16;
    *(_DWORD *)(v11 + 272) = 0;
    *(_BYTE *)(v11 + 408) = 0;
    *(_QWORD *)(v11 + 416) = a2;
    *(_QWORD *)(v11 + 424) = v7;
    *(_QWORD *)(v11 + 432) = v10;
  }
  v12 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 160) = v11;
  if ( v12 )
  {
    v13 = *(_QWORD *)(v12 + 256);
    if ( v13 != *(_QWORD *)(v12 + 248) )
      _libc_free(v13);
    v14 = *(_QWORD *)(v12 + 88);
    if ( v14 != *(_QWORD *)(v12 + 80) )
      _libc_free(v14);
    j___libc_free_0(*(_QWORD *)(v12 + 40));
    if ( *(_DWORD *)(v12 + 24) )
    {
      v21 = -8;
      v22 = 0;
      v25 = -16;
      v26 = 0;
      v23 = &unk_49E8A80;
      v15 = *(unsigned int *)(v12 + 24);
      v20[0] = 2;
      v20[1] = 0;
      v24[0] = 2;
      v24[1] = 0;
      v16 = *(_QWORD **)(v12 + 8);
      v17 = &v16[5 * v15];
      if ( v16 != v17 )
      {
        do
        {
          v18 = v16[3];
          *v16 = &unk_49EE2B0;
          if ( v18 != -8 && v18 != 0 && v18 != -16 )
            sub_1649B30(v16 + 1);
          v16 += 5;
        }
        while ( v17 != v16 );
        v23 = &unk_49EE2B0;
        if ( v25 != 0 && v25 != -16 && v25 != -8 )
          sub_1649B30(v24);
      }
      if ( v21 != 0 && v21 != -8 && v21 != -16 )
        sub_1649B30(v20);
    }
    j___libc_free_0(*(_QWORD *)(v12 + 8));
    j_j___libc_free_0(v12, 440);
  }
  return 0;
}
