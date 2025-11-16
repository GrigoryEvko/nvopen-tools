// Function: sub_3205740
// Address: 0x3205740
//
__int64 __fastcall sub_3205740(__int64 a1, __int64 a2, unsigned __int8 *a3)
{
  __int64 v4; // r13
  const char *v5; // rax
  char *v6; // rdx
  __int64 v7; // r9
  char *v8; // r8
  char *v9; // rcx
  char *v11; // rdx

  v4 = sub_AF2660(a3);
  v5 = sub_AF5A10(a3, a2);
  if ( v6 )
  {
    v8 = v6;
    v9 = (char *)v5;
  }
  else
  {
    v9 = (char *)sub_31F3D90((__int64)a3);
    v8 = v11;
  }
  sub_3205680(a1, a2, v4, v9, v8, v7);
  return a1;
}
