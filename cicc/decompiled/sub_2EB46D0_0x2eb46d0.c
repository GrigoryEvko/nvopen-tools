// Function: sub_2EB46D0
// Address: 0x2eb46d0
//
void __fastcall sub_2EB46D0(__int64 a1)
{
  __int64 v1; // rax
  _QWORD *v2; // rbx
  __int64 v3; // r12
  _QWORD *v4; // r12
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v2 = (_QWORD *)(a1 + 16);
    v3 = 36;
  }
  else
  {
    v1 = *(unsigned int *)(a1 + 24);
    if ( !(_DWORD)v1 )
      return;
    v2 = *(_QWORD **)(a1 + 16);
    v3 = 9 * v1;
  }
  v4 = &v2[v3];
  do
  {
    if ( *v2 != -8192 && *v2 != -4096 )
    {
      v5 = v2[5];
      if ( (_QWORD *)v5 != v2 + 7 )
        _libc_free(v5);
      v6 = v2[1];
      if ( (_QWORD *)v6 != v2 + 3 )
        _libc_free(v6);
    }
    v2 += 9;
  }
  while ( v2 != v4 );
}
