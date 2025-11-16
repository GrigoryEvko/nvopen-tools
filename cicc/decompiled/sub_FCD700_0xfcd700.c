// Function: sub_FCD700
// Address: 0xfcd700
//
__int64 __fastcall sub_FCD700(_QWORD *a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 v4; // rsi
  _QWORD *v5; // rbx
  __int64 v6; // rsi
  _QWORD *v7; // r12
  __int64 v8; // r15
  __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // rdi

  v3 = a1[22];
  *a1 = &unk_49E51E8;
  if ( v3 )
  {
    if ( !*(_BYTE *)(v3 + 204) )
      _libc_free(*(_QWORD *)(v3 + 184), a2);
    sub_C7D6A0(*(_QWORD *)(v3 + 152), 16LL * *(unsigned int *)(v3 + 168), 8);
    v4 = *(unsigned int *)(v3 + 136);
    if ( (_DWORD)v4 )
    {
      v5 = *(_QWORD **)(v3 + 120);
      v6 = 2 * v4;
      v7 = &v5[v6];
      do
      {
        if ( *v5 != -8192 && *v5 != -4096 )
        {
          v8 = v5[1];
          if ( v8 )
          {
            v9 = *(_QWORD *)(v8 + 96);
            if ( v9 != v8 + 112 )
              _libc_free(v9, v6 * 8);
            v10 = *(_QWORD *)(v8 + 24);
            if ( v10 != v8 + 40 )
              _libc_free(v10, v6 * 8);
            v6 = 21;
            j_j___libc_free_0(v8, 168);
          }
        }
        v5 += 2;
      }
      while ( v7 != v5 );
      v4 = *(unsigned int *)(v3 + 136);
    }
    sub_C7D6A0(*(_QWORD *)(v3 + 120), 16 * v4, 8);
    v11 = *(_QWORD *)(v3 + 88);
    if ( v11 )
      j_j___libc_free_0(v11, *(_QWORD *)(v3 + 104) - v11);
    sub_C7D6A0(*(_QWORD *)(v3 + 64), 16LL * *(unsigned int *)(v3 + 80), 8);
    j_j___libc_free_0(v3, 272);
  }
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  return j_j___libc_free_0(a1, 184);
}
