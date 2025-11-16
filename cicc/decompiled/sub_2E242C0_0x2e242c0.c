// Function: sub_2E242C0
// Address: 0x2e242c0
//
void __fastcall sub_2E242C0(__int64 a1)
{
  _QWORD **v2; // r13
  _QWORD **v3; // r12
  unsigned __int64 v4; // rdi
  _QWORD *v5; // rbx
  unsigned __int64 v6; // rdi

  v2 = *(_QWORD ***)(a1 + 200);
  v3 = &v2[7 * *(unsigned int *)(a1 + 208)];
  while ( v2 != v3 )
  {
    while ( 1 )
    {
      v4 = (unsigned __int64)*(v3 - 3);
      v3 -= 7;
      if ( v4 )
        j_j___libc_free_0(v4);
      v5 = *v3;
      if ( v3 == *v3 )
        break;
      do
      {
        v6 = (unsigned __int64)v5;
        v5 = (_QWORD *)*v5;
        j_j___libc_free_0(v6);
      }
      while ( v3 != v5 );
      if ( v2 == v3 )
        goto LABEL_8;
    }
  }
LABEL_8:
  *(_DWORD *)(a1 + 208) = 0;
}
