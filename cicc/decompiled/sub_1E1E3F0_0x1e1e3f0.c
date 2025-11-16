// Function: sub_1E1E3F0
// Address: 0x1e1e3f0
//
__int64 __fastcall sub_1E1E3F0(__int64 a1)
{
  __int64 v2; // rbx
  __int64 v3; // rdi
  unsigned __int64 *v4; // r13
  __int64 v5; // rax
  unsigned __int64 *v6; // rbx

  v2 = *(_QWORD *)(a1 + 904);
  *(_DWORD *)(a1 + 752) = 0;
  while ( v2 )
  {
    sub_1E1D070(*(_QWORD *)(v2 + 24));
    v3 = v2;
    v2 = *(_QWORD *)(v2 + 16);
    j_j___libc_free_0(v3, 40);
  }
  v4 = *(unsigned __int64 **)(a1 + 1032);
  *(_QWORD *)(a1 + 904) = 0;
  *(_QWORD *)(a1 + 912) = a1 + 896;
  *(_QWORD *)(a1 + 920) = a1 + 896;
  v5 = *(unsigned int *)(a1 + 1040);
  *(_QWORD *)(a1 + 928) = 0;
  *(_DWORD *)(a1 + 944) = 0;
  *(_DWORD *)(a1 + 992) = 0;
  v6 = &v4[6 * v5];
  while ( v4 != v6 )
  {
    while ( 1 )
    {
      v6 -= 6;
      if ( (unsigned __int64 *)*v6 == v6 + 2 )
        break;
      _libc_free(*v6);
      if ( v4 == v6 )
        goto LABEL_7;
    }
  }
LABEL_7:
  *(_DWORD *)(a1 + 1040) = 0;
  return sub_1E1E220(a1 + 1816);
}
