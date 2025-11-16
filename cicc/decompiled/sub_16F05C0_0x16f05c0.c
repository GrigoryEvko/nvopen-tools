// Function: sub_16F05C0
// Address: 0x16f05c0
//
void __fastcall sub_16F05C0(__int64 a1)
{
  unsigned __int64 v1; // r12
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  if ( *(_DWORD *)a1 == 62053 )
  {
    v1 = *(_QWORD *)(a1 + 24);
    if ( v1 )
    {
      if ( *(_DWORD *)v1 == 53829 )
      {
        *(_DWORD *)a1 = 0;
        v2 = *(_QWORD *)(v1 + 8);
        if ( v2 )
          _libc_free(v2);
        v3 = *(_QWORD *)(v1 + 24);
        if ( v3 )
          _libc_free(v3);
        v4 = *(_QWORD *)(v1 + 32);
        if ( v4 )
          _libc_free(v4);
        v5 = *(_QWORD *)(v1 + 96);
        if ( v5 )
          _libc_free(v5);
        _libc_free(v1);
      }
    }
  }
}
