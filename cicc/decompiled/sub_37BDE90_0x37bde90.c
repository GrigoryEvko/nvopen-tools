// Function: sub_37BDE90
// Address: 0x37bde90
//
void __fastcall sub_37BDE90(__int64 a1)
{
  __int64 v1; // rax
  _QWORD *v2; // rbx
  _QWORD *v3; // r12
  unsigned __int64 v4; // rdi

  v1 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v1 )
  {
    v2 = *(_QWORD **)(a1 + 8);
    v3 = &v2[7 * v1];
    while ( *v2 == -4096 )
    {
      if ( v2[1] == -1 && v2[2] == -1 )
      {
        v2 += 7;
        if ( v3 == v2 )
          return;
      }
      else
      {
LABEL_4:
        v4 = v2[3];
        if ( (_QWORD *)v4 != v2 + 5 )
          _libc_free(v4);
LABEL_6:
        v2 += 7;
        if ( v3 == v2 )
          return;
      }
    }
    if ( *v2 == -8192 && v2[1] == -2 && v2[2] == -2 )
      goto LABEL_6;
    goto LABEL_4;
  }
}
