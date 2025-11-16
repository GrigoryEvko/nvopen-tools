// Function: sub_1568D80
// Address: 0x1568d80
//
__int64 __fastcall sub_1568D80(__int64 a1, void **a2)
{
  unsigned int v2; // eax
  _DWORD *v3; // rbx
  unsigned int v4; // r13d
  unsigned int v5; // r12d
  __int64 v7; // rax

  *a2 = 0;
  v2 = sub_15644B0(a1, a2);
  v3 = *a2;
  v4 = v2;
  if ( !*a2 )
    v3 = (_DWORD *)a1;
  v5 = v3[9];
  if ( v5 )
  {
    v7 = sub_15E0530(v3);
    *((_QWORD *)v3 + 14) = sub_15E1850(v7, v5);
  }
  return v4;
}
