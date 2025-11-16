// Function: sub_157EE90
// Address: 0x157ee90
//
void __fastcall sub_157EE90(__int64 a1)
{
  __int64 v1; // r8
  __int64 i; // r9
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *v5; // rdi
  __int64 v6; // rcx
  unsigned __int64 v7; // rdx

  v1 = *(_QWORD *)(a1 + 48);
  for ( i = a1 + 40; i != v1; v1 = *(_QWORD *)(v1 + 8) )
  {
    if ( !v1 )
      BUG();
    v3 = 3LL * (*(_DWORD *)(v1 - 4) & 0xFFFFFFF);
    if ( (*(_BYTE *)(v1 - 1) & 0x40) != 0 )
    {
      v4 = *(_QWORD **)(v1 - 32);
      v5 = &v4[v3];
    }
    else
    {
      v5 = (_QWORD *)(v1 - 24);
      v4 = (_QWORD *)(v1 - 24 - v3 * 8);
    }
    for ( ; v5 != v4; v4 += 3 )
    {
      if ( *v4 )
      {
        v6 = v4[1];
        v7 = v4[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v7 = v6;
        if ( v6 )
          *(_QWORD *)(v6 + 16) = *(_QWORD *)(v6 + 16) & 3LL | v7;
      }
      *v4 = 0;
    }
  }
}
