// Function: sub_12BAA30
// Address: 0x12baa30
//
__int64 __fastcall sub_12BAA30(_DWORD *a1, _DWORD *a2, _DWORD *a3, _DWORD *a4)
{
  char v7; // r15
  __int64 v8; // rdi
  __int64 v10; // [rsp+8h] [rbp-38h]

  v7 = byte_4F92D70;
  if ( byte_4F92D70 || !dword_4C6F008 )
  {
    if ( !qword_4F92D80 )
      sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
    v7 = 1;
    v10 = qword_4F92D80;
    sub_16C30C0(qword_4F92D80);
    v8 = v10;
  }
  else
  {
    if ( !qword_4F92D80 )
      sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
    v8 = qword_4F92D80;
  }
  if ( a1 )
    *a1 = 2;
  if ( a2 )
    *a2 = 0;
  if ( a3 )
    *a3 = 3;
  if ( a4 )
    *a4 = 2;
  if ( v7 )
    sub_16C30E0(v8);
  return 0;
}
