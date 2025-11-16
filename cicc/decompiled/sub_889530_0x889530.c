// Function: sub_889530
// Address: 0x889530
//
void __fastcall sub_889530(unsigned __int8 *a1)
{
  unsigned __int8 *v1; // rbx
  unsigned __int8 v2; // di
  __int64 v3; // r13
  _QWORD *v4; // r12
  _QWORD *v5; // rax
  _QWORD *v6; // rax

  v1 = a1;
  v2 = *a1;
  if ( v2 != 14 )
  {
    do
    {
      v3 = *((_QWORD *)v1 + 1);
      v1 += 32;
      v4 = sub_72C610(v2);
      v5 = sub_72B5A0((__int64)v4, v3, 2);
      sub_888EB0(*((char **)v1 - 2), (__int64)v5);
      v6 = sub_72B5A0((__int64)v4, 2 * v3, 2);
      sub_888EB0(*((char **)v1 - 1), (__int64)v6);
      v2 = *v1;
    }
    while ( *v1 != 14 );
  }
}
