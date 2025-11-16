// Function: sub_888F20
// Address: 0x888f20
//
void __fastcall sub_888F20(unsigned __int8 *a1)
{
  unsigned __int8 *v1; // rbx
  unsigned __int8 v2; // di
  __int64 v3; // r12
  _QWORD *v4; // rax
  _QWORD *v5; // rax

  v1 = a1;
  v2 = *a1;
  if ( v2 != 13 )
  {
    do
    {
      v3 = *((_QWORD *)v1 + 1);
      v1 += 24;
      v4 = sub_72BA30(v2);
      v5 = sub_72B5A0((__int64)v4, v3, 4);
      sub_888EB0(*((char **)v1 - 1), (__int64)v5);
      v2 = *v1;
    }
    while ( *v1 != 13 );
  }
}
