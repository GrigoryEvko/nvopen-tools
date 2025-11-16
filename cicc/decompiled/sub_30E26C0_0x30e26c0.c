// Function: sub_30E26C0
// Address: 0x30e26c0
//
void __fastcall sub_30E26C0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r13
  bool v5; // cc
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  void (__fastcall *v8)(__int64, __int64, __int64); // rax
  unsigned __int64 v9; // rdi

  *(_QWORD *)a1 = off_49D8B58;
  v2 = *(unsigned int *)(a1 + 240);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD *)(a1 + 224);
    v4 = v3 + 56 * v2;
    do
    {
      while ( 1 )
      {
        if ( *(_QWORD *)v3 != -8192 && *(_QWORD *)v3 != -4096 && *(_BYTE *)(v3 + 48) )
        {
          v5 = *(_DWORD *)(v3 + 40) <= 0x40u;
          *(_BYTE *)(v3 + 48) = 0;
          if ( !v5 )
          {
            v6 = *(_QWORD *)(v3 + 32);
            if ( v6 )
              j_j___libc_free_0_0(v6);
          }
          if ( *(_DWORD *)(v3 + 24) > 0x40u )
          {
            v7 = *(_QWORD *)(v3 + 16);
            if ( v7 )
              break;
          }
        }
        v3 += 56;
        if ( v4 == v3 )
          goto LABEL_13;
      }
      j_j___libc_free_0_0(v7);
      v3 += 56;
    }
    while ( v4 != v3 );
LABEL_13:
    v2 = *(unsigned int *)(a1 + 240);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 224), 56 * v2, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 192), 16LL * *(unsigned int *)(a1 + 208), 8);
  v8 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 168);
  if ( v8 )
    v8(a1 + 152, a1 + 152, 3);
  v9 = *(_QWORD *)(a1 + 8);
  if ( v9 != a1 + 24 )
    _libc_free(v9);
}
