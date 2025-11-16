// Function: sub_3509890
// Address: 0x3509890
//
void __fastcall sub_3509890(__int64 a1)
{
  __int64 v1; // r12
  _QWORD *v2; // rsi
  _QWORD *v3; // rdx
  _QWORD *v4; // rax
  __int64 v5; // rcx
  __int64 *v6; // rax

  v1 = *(_QWORD *)(a1 + 24);
  *(_QWORD *)a1 = &unk_4A388F0;
  if ( *(_BYTE *)(v1 + 36) )
  {
    v2 = *(_QWORD **)(v1 + 16);
    v3 = &v2[*(unsigned int *)(v1 + 28)];
    v4 = v2;
    if ( v2 != v3 )
    {
      while ( a1 != *v4 )
      {
        if ( v3 == ++v4 )
          goto LABEL_7;
      }
      v5 = (unsigned int)(*(_DWORD *)(v1 + 28) - 1);
      *(_DWORD *)(v1 + 28) = v5;
      *v4 = v2[v5];
      ++*(_QWORD *)(v1 + 8);
    }
LABEL_7:
    if ( *(_BYTE *)(a1 + 172) )
      goto LABEL_8;
    goto LABEL_12;
  }
  v6 = sub_C8CA60(v1 + 8, a1);
  if ( !v6 )
    goto LABEL_7;
  *v6 = -2;
  ++*(_DWORD *)(v1 + 32);
  ++*(_QWORD *)(v1 + 8);
  if ( *(_BYTE *)(a1 + 172) )
  {
LABEL_8:
    if ( *(_BYTE *)(a1 + 108) )
      return;
LABEL_13:
    _libc_free(*(_QWORD *)(a1 + 88));
    return;
  }
LABEL_12:
  _libc_free(*(_QWORD *)(a1 + 152));
  if ( !*(_BYTE *)(a1 + 108) )
    goto LABEL_13;
}
