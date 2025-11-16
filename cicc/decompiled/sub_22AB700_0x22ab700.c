// Function: sub_22AB700
// Address: 0x22ab700
//
__int64 __fastcall sub_22AB700(_QWORD *a1)
{
  unsigned __int64 v1; // r13
  unsigned __int64 *v2; // r15
  unsigned __int64 v3; // rcx
  __int64 v4; // rcx
  unsigned __int64 *v5; // rbx
  unsigned __int64 v6; // rsi
  bool v7; // zf

  v1 = a1[22];
  *a1 = &unk_4A09C08;
  if ( v1 )
  {
    if ( !*(_BYTE *)(v1 + 244) )
      _libc_free(*(_QWORD *)(v1 + 224));
    v2 = *(unsigned __int64 **)(v1 + 208);
    while ( (unsigned __int64 *)(v1 + 200) != v2 )
    {
      v5 = v2;
      v2 = (unsigned __int64 *)v2[1];
      v6 = *v5 & 0xFFFFFFFFFFFFFFF8LL;
      *v2 = v6 | *v2 & 7;
      *(_QWORD *)(v6 + 8) = v2;
      *v5 &= 7u;
      v7 = *((_BYTE *)v5 + 76) == 0;
      v5[1] = 0;
      *(v5 - 4) = (unsigned __int64)&unk_4A09CC0;
      if ( v7 )
        _libc_free(v5[7]);
      v3 = v5[5];
      if ( v3 != 0 && v3 != -4096 && v3 != -8192 )
        sub_BD60C0(v5 + 3);
      *(v5 - 4) = (unsigned __int64)&unk_49DB368;
      v4 = *(v5 - 1);
      if ( v4 != 0 && v4 != -4096 && v4 != -8192 )
        sub_BD60C0(v5 - 3);
      j_j___libc_free_0((unsigned __int64)(v5 - 4));
    }
    if ( !*(_BYTE *)(v1 + 68) )
      _libc_free(*(_QWORD *)(v1 + 48));
    j_j___libc_free_0(v1);
  }
  *a1 = &unk_49DE2C8;
  return sub_BB9100((__int64)a1);
}
