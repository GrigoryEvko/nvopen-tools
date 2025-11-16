// Function: sub_13C2BD0
// Address: 0x13c2bd0
//
void __fastcall sub_13C2BD0(__int64 a1)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // r12
  __int64 v4; // rax
  __int64 v5; // r12
  _QWORD *v6; // rbx
  _QWORD *v7; // r12
  unsigned __int64 v8; // r14
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi

  v2 = *(_QWORD **)(a1 + 328);
  while ( v2 != (_QWORD *)(a1 + 328) )
  {
    v3 = v2;
    v2 = (_QWORD *)*v2;
    v4 = v3[5];
    v3[2] = &unk_49EE2B0;
    if ( v4 != 0 && v4 != -8 && v4 != -16 )
      sub_1649B30(v3 + 3);
    j_j___libc_free_0(v3, 64);
  }
  j___libc_free_0(*(_QWORD *)(a1 + 304));
  v5 = *(unsigned int *)(a1 + 288);
  if ( (_DWORD)v5 )
  {
    v6 = *(_QWORD **)(a1 + 272);
    v7 = &v6[2 * v5];
    do
    {
      while ( 1 )
      {
        if ( *v6 != -8 && *v6 != -16 )
        {
          v8 = v6[1] & 0xFFFFFFFFFFFFFFF8LL;
          if ( v8 )
            break;
        }
        v6 += 2;
        if ( v7 == v6 )
          goto LABEL_15;
      }
      if ( (*(_BYTE *)(v8 + 8) & 1) != 0 )
      {
        j_j___libc_free_0(v6[1] & 0xFFFFFFFFFFFFFFF8LL, 272);
      }
      else
      {
        j___libc_free_0(*(_QWORD *)(v8 + 16));
        j_j___libc_free_0(v8, 272);
      }
      v6 += 2;
    }
    while ( v7 != v6 );
  }
LABEL_15:
  j___libc_free_0(*(_QWORD *)(a1 + 272));
  j___libc_free_0(*(_QWORD *)(a1 + 240));
  v9 = *(_QWORD *)(a1 + 144);
  if ( v9 != *(_QWORD *)(a1 + 136) )
    _libc_free(v9);
  v10 = *(_QWORD *)(a1 + 40);
  if ( v10 != *(_QWORD *)(a1 + 32) )
    _libc_free(v10);
}
