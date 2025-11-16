// Function: sub_21F1D90
// Address: 0x21f1d90
//
__int64 __fastcall sub_21F1D90(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // r14
  __int64 v20; // rax
  _QWORD *v21; // rbx
  _QWORD *v22; // rdx
  __int64 v23; // r15
  __int64 v24; // rdi
  _QWORD *v26; // [rsp+8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_29:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4FC6A0C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_29;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4FC6A0C);
  v7 = *(__int64 **)(a1 + 8);
  v8 = v6;
  v9 = *v7;
  v10 = v7[1];
  if ( v9 == v10 )
LABEL_28:
    BUG();
  while ( *(_UNKNOWN **)v9 != &unk_4FC62EC )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_28;
  }
  v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4FC62EC);
  v12 = sub_22077B0(328);
  v13 = v12;
  if ( v12 )
  {
    v14 = *(_QWORD *)(a2 + 16);
    *(_QWORD *)v12 = a2;
    *(_QWORD *)(v12 + 8) = v8;
    *(_QWORD *)(v12 + 16) = v11;
    *(_QWORD *)(v12 + 24) = 0;
    *(_QWORD *)(v12 + 32) = 0;
    *(_QWORD *)(v12 + 40) = 0;
    *(_BYTE *)(v12 + 48) = 0;
    *(_QWORD *)(v12 + 56) = 0;
    *(_QWORD *)(v12 + 64) = 0;
    *(_QWORD *)(v12 + 72) = 0;
    *(_DWORD *)(v12 + 80) = 0;
    *(_QWORD *)(v12 + 88) = 0;
    *(_QWORD *)(v12 + 96) = 0;
    *(_QWORD *)(v12 + 104) = 0;
    *(_QWORD *)(v12 + 112) = 0;
    *(_QWORD *)(v12 + 120) = 0;
    *(_QWORD *)(v12 + 128) = 0;
    *(_DWORD *)(v12 + 136) = 0;
    *(_QWORD *)(v12 + 144) = 0;
    *(_QWORD *)(v12 + 152) = 0;
    *(_QWORD *)(v12 + 160) = 0;
    *(_DWORD *)(v12 + 168) = 0;
    v15 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v14 + 112LL))(v14);
    v16 = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(v13 + 176) = v15;
    v17 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v16 + 40LL))(v16);
    *(_QWORD *)(v13 + 200) = 0;
    *(_QWORD *)(v13 + 184) = v17;
    v18 = *(_QWORD *)(a2 + 40);
    *(_QWORD *)(v13 + 208) = 0;
    *(_QWORD *)(v13 + 192) = v18;
    *(_QWORD *)(v13 + 216) = 0;
    *(_DWORD *)(v13 + 224) = 0;
    *(_QWORD *)(v13 + 232) = 0;
    *(_QWORD *)(v13 + 240) = 0;
    *(_QWORD *)(v13 + 248) = 0;
    *(_DWORD *)(v13 + 256) = 0;
    *(_QWORD *)(v13 + 264) = 0;
    *(_QWORD *)(v13 + 272) = 0;
    *(_QWORD *)(v13 + 280) = 0;
    *(_DWORD *)(v13 + 288) = 0;
    *(_QWORD *)(v13 + 296) = 0;
    *(_QWORD *)(v13 + 304) = 0;
    *(_QWORD *)(v13 + 312) = 0;
    *(_DWORD *)(v13 + 320) = 0;
  }
  v19 = *(_QWORD *)(a1 + 232);
  *(_QWORD *)(a1 + 232) = v13;
  if ( v19 )
  {
    j___libc_free_0(*(_QWORD *)(v19 + 304));
    j___libc_free_0(*(_QWORD *)(v19 + 272));
    j___libc_free_0(*(_QWORD *)(v19 + 240));
    j___libc_free_0(*(_QWORD *)(v19 + 208));
    j___libc_free_0(*(_QWORD *)(v19 + 152));
    v20 = *(unsigned int *)(v19 + 136);
    if ( (_DWORD)v20 )
    {
      v21 = *(_QWORD **)(v19 + 120);
      v22 = &v21[2 * v20];
      do
      {
        if ( *v21 != -16 && *v21 != -8 )
        {
          v23 = v21[1];
          if ( v23 )
          {
            v26 = v22;
            _libc_free(*(_QWORD *)(v23 + 48));
            _libc_free(*(_QWORD *)(v23 + 24));
            j_j___libc_free_0(v23, 72);
            v22 = v26;
          }
        }
        v21 += 2;
      }
      while ( v22 != v21 );
    }
    j___libc_free_0(*(_QWORD *)(v19 + 120));
    v24 = *(_QWORD *)(v19 + 88);
    if ( v24 )
      j_j___libc_free_0(v24, *(_QWORD *)(v19 + 104) - v24);
    j___libc_free_0(*(_QWORD *)(v19 + 64));
    j_j___libc_free_0(v19, 328);
  }
  if ( (**(_BYTE **)(**(_QWORD **)(a2 + 40) + 352LL) & 1) != 0 )
    sub_21F15B0(*(_QWORD *)(a1 + 232));
  return 0;
}
