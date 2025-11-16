// Function: sub_139F320
// Address: 0x139f320
//
__int64 __fastcall sub_139F320(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 *v8; // rdx
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r15
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 v16; // rdx
  __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  __int64 v19; // [rsp+8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_24:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F9D764 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_24;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F9D764);
  v7 = sub_14CF090(v6, a2);
  v8 = *(__int64 **)(a1 + 8);
  v9 = v7;
  v10 = *v8;
  v11 = v8[1];
  if ( v10 == v11 )
LABEL_25:
    BUG();
  while ( *(_UNKNOWN **)v10 != &unk_4F9E06C )
  {
    v10 += 16;
    if ( v11 == v10 )
      goto LABEL_25;
  }
  v12 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v10 + 8) + 104LL))(*(_QWORD *)(v10 + 8), &unk_4F9E06C)
      + 160;
  if ( *(_BYTE *)(a1 + 520) )
  {
    v14 = *(unsigned int *)(a1 + 512);
    if ( (_DWORD)v14 )
    {
      v15 = *(_QWORD *)(a1 + 496);
      v16 = v15 + 24 * v14;
      do
      {
        if ( *(_QWORD *)v15 != -8 && *(_QWORD *)v15 != -16 && *(_DWORD *)(v15 + 16) > 0x40u )
        {
          v17 = *(_QWORD *)(v15 + 8);
          if ( v17 )
          {
            v19 = v16;
            j_j___libc_free_0_0(v17);
            v16 = v19;
          }
        }
        v15 += 24;
      }
      while ( v16 != v15 );
    }
    j___libc_free_0(*(_QWORD *)(a1 + 496));
    v18 = *(_QWORD *)(a1 + 208);
    if ( v18 != *(_QWORD *)(a1 + 200) )
      _libc_free(v18);
  }
  *(_QWORD *)(a1 + 160) = a2;
  *(_QWORD *)(a1 + 200) = a1 + 232;
  *(_QWORD *)(a1 + 208) = a1 + 232;
  *(_BYTE *)(a1 + 520) = 1;
  *(_QWORD *)(a1 + 168) = v9;
  *(_QWORD *)(a1 + 176) = v12;
  *(_BYTE *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 216) = 32;
  *(_DWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 488) = 0;
  *(_QWORD *)(a1 + 496) = 0;
  *(_QWORD *)(a1 + 504) = 0;
  *(_DWORD *)(a1 + 512) = 0;
  return 0;
}
