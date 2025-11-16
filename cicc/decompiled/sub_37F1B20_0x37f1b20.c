// Function: sub_37F1B20
// Address: 0x37f1b20
//
__int64 __fastcall sub_37F1B20(__int64 a1)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 v6; // r12
  unsigned __int64 v7; // rdi

  *(_QWORD *)a1 = &unk_4A3D858;
  v2 = *(_QWORD *)(a1 + 232);
  if ( v2 != a1 + 248 )
    _libc_free(v2);
  v3 = *(unsigned int *)(a1 + 224);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD *)(a1 + 208);
    v5 = v4 + 56 * v3;
    do
    {
      v6 = v4 + 56;
      if ( *(_QWORD *)v4 != -8192 && *(_QWORD *)v4 != -4096 )
      {
        v7 = *(_QWORD *)(v4 + 40);
        if ( v7 != v6 )
          _libc_free(v7);
        sub_C7D6A0(*(_QWORD *)(v4 + 16), 8LL * *(unsigned int *)(v4 + 32), 8);
      }
      v4 += 56;
    }
    while ( v5 != v6 );
    v3 = *(unsigned int *)(a1 + 224);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 208), 56 * v3, 8);
  *(_QWORD *)a1 = &unk_49DAF80;
  return sub_BB9100(a1);
}
