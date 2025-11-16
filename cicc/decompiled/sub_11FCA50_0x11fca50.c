// Function: sub_11FCA50
// Address: 0x11fca50
//
__int64 __fastcall sub_11FCA50(__int64 a1, __int64 a2)
{
  __int64 *v3; // rbx
  __int64 *v4; // r13
  __int64 v5; // r12
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rsi
  __int64 v9; // rdi

  v3 = *(__int64 **)(a1 + 264);
  v4 = *(__int64 **)(a1 + 256);
  *(_QWORD *)a1 = &unk_49E6540;
  if ( v3 != v4 )
  {
    do
    {
      v5 = *v4;
      if ( *v4 )
      {
        v6 = *(_QWORD *)(v5 + 176);
        if ( v6 != v5 + 192 )
          _libc_free(v6, a2);
        v7 = *(_QWORD *)(v5 + 88);
        if ( v7 != v5 + 104 )
          _libc_free(v7, a2);
        v8 = 8LL * *(unsigned int *)(v5 + 80);
        sub_C7D6A0(*(_QWORD *)(v5 + 64), v8, 8);
        sub_11FC810(v5 + 32, v8);
        v9 = *(_QWORD *)(v5 + 8);
        if ( v9 != v5 + 24 )
          _libc_free(v9, v8);
        a2 = 224;
        j_j___libc_free_0(v5, 224);
      }
      ++v4;
    }
    while ( v3 != v4 );
    v4 = *(__int64 **)(a1 + 256);
  }
  if ( v4 )
    j_j___libc_free_0(v4, *(_QWORD *)(a1 + 272) - (_QWORD)v4);
  sub_C7D6A0(*(_QWORD *)(a1 + 232), 16LL * *(unsigned int *)(a1 + 248), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 200), 16LL * *(unsigned int *)(a1 + 216), 8);
  *(_QWORD *)a1 = &unk_49DAF80;
  sub_BB9100(a1);
  return j_j___libc_free_0(a1, 280);
}
