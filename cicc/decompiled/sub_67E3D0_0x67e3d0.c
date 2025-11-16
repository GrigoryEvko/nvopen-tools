// Function: sub_67E3D0
// Address: 0x67e3d0
//
__int64 __fastcall sub_67E3D0(_QWORD *a1)
{
  _QWORD *v1; // rcx
  _QWORD *v3; // rdi
  __int64 result; // rax
  _QWORD *v5; // rcx
  _QWORD *v6; // r8

  v1 = (_QWORD *)*a1;
  if ( *a1 )
  {
    while ( 1 )
    {
      v3 = v1;
      v1 = (_QWORD *)v1[1];
      if ( dword_4D03A00 != -1 )
        break;
      if ( !v1 )
        goto LABEL_7;
    }
    while ( 1 )
    {
      result = sub_67C610(v3);
      if ( !v5 )
        break;
      v3 = v5;
    }
    *v6 = 0;
    v6[1] = 0;
  }
  else
  {
LABEL_7:
    *a1 = 0;
    a1[1] = 0;
  }
  return result;
}
