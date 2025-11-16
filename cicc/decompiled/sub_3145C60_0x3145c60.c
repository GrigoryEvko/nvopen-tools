// Function: sub_3145C60
// Address: 0x3145c60
//
void __fastcall sub_3145C60(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi
  __int64 v5; // rax
  unsigned __int64 v6; // r12

  v2 = sub_22077B0(0x30u);
  if ( v2 )
  {
    *(_QWORD *)(v2 + 40) = 0;
    *(_QWORD *)(v2 + 32) = v2 + 48;
    *(_OWORD *)v2 = 0;
    *(_OWORD *)(v2 + 16) = 0;
  }
  v3 = *(_QWORD *)(a1 + 48);
  *(_QWORD *)(a1 + 48) = v2;
  if ( v3 )
  {
    v4 = *(_QWORD *)(v3 + 32);
    if ( v4 != v3 + 48 )
      _libc_free(v4);
    sub_C7D6A0(*(_QWORD *)(v3 + 8), 8LL * *(unsigned int *)(v3 + 24), 4);
    j_j___libc_free_0(v3);
  }
  v5 = sub_22077B0(0x20u);
  if ( v5 )
  {
    *(_QWORD *)v5 = 0;
    *(_QWORD *)(v5 + 8) = 0;
    *(_QWORD *)(v5 + 16) = 0;
    *(_DWORD *)(v5 + 24) = 0;
  }
  v6 = *(_QWORD *)(a1 + 56);
  *(_QWORD *)(a1 + 56) = v5;
  if ( v6 )
  {
    sub_C7D6A0(*(_QWORD *)(v6 + 8), 16LL * *(unsigned int *)(v6 + 24), 8);
    j_j___libc_free_0(v6);
  }
}
