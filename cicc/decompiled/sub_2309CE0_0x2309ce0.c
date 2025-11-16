// Function: sub_2309CE0
// Address: 0x2309ce0
//
void __fastcall sub_2309CE0(__int64 a1)
{
  bool v1; // zf
  unsigned __int64 *v2; // r15
  unsigned __int64 v3; // rdx
  __int64 v4; // rdx
  unsigned __int64 *v5; // rbx
  unsigned __int64 v6; // rcx

  v1 = *(_BYTE *)(a1 + 252) == 0;
  *(_QWORD *)a1 = &unk_4A0AC78;
  if ( v1 )
    _libc_free(*(_QWORD *)(a1 + 232));
  v2 = *(unsigned __int64 **)(a1 + 216);
  while ( (unsigned __int64 *)(a1 + 208) != v2 )
  {
    v5 = v2;
    v2 = (unsigned __int64 *)v2[1];
    v6 = *v5 & 0xFFFFFFFFFFFFFFF8LL;
    *v2 = v6 | *v2 & 7;
    *(_QWORD *)(v6 + 8) = v2;
    *v5 &= 7u;
    v1 = *((_BYTE *)v5 + 76) == 0;
    v5[1] = 0;
    *(v5 - 4) = (unsigned __int64)&unk_4A09CC0;
    if ( v1 )
      _libc_free(v5[7]);
    v3 = v5[5];
    if ( v3 != 0 && v3 != -4096 && v3 != -8192 )
      sub_BD60C0(v5 + 3);
    *(v5 - 4) = (unsigned __int64)&unk_49DB368;
    v4 = *(v5 - 1);
    if ( v4 != -4096 && v4 != 0 && v4 != -8192 )
      sub_BD60C0(v5 - 3);
    j_j___libc_free_0((unsigned __int64)(v5 - 4));
  }
  if ( !*(_BYTE *)(a1 + 76) )
    _libc_free(*(_QWORD *)(a1 + 56));
}
