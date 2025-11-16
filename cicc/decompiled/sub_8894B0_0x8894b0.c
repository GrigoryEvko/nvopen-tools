// Function: sub_8894B0
// Address: 0x8894b0
//
void __fastcall sub_8894B0(unsigned __int8 *a1, char a2)
{
  unsigned __int8 *v2; // rbx
  unsigned __int8 v3; // di
  __int64 v4; // r12
  _QWORD *v5; // r14
  _QWORD *v6; // rax
  _QWORD *v7; // rax

  v2 = a1;
  v3 = *a1;
  if ( v3 != 13 )
  {
    do
    {
      v4 = *((_QWORD *)v2 + 1);
      v2 += 32;
      v5 = sub_72BA30(v3);
      v6 = sub_72B5A0((__int64)v5, v4, a2);
      sub_888EB0(*((char **)v2 - 2), (__int64)v6);
      v7 = sub_72B5A0((__int64)v5, 2 * v4, a2);
      sub_888EB0(*((char **)v2 - 1), (__int64)v7);
      v3 = *v2;
    }
    while ( *v2 != 13 );
  }
}
