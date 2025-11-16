// Function: sub_C3D480
// Address: 0xc3d480
//
__int64 __fastcall sub_C3D480(__int64 a1, unsigned __int8 a2, unsigned __int8 a3, unsigned __int64 *a4)
{
  unsigned int v5; // r14d
  void *v6; // rbx
  void **v7; // rdi
  void **v9; // [rsp+8h] [rbp-38h]

  v5 = a3;
  v9 = *(void ***)(a1 + 8);
  v6 = sub_C33340();
  if ( *v9 == v6 )
    sub_C3D480(v9, a2, v5, a4);
  else
    sub_C36070((__int64)v9, a2, v5, a4);
  v7 = (void **)(*(_QWORD *)(a1 + 8) + 24LL);
  if ( *v7 == v6 )
    return sub_C3CEB0(v7, 0);
  else
    return sub_C37310((__int64)v7, 0);
}
