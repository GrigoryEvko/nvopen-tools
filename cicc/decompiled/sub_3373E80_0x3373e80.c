// Function: sub_3373E80
// Address: 0x3373e80
//
bool __fastcall sub_3373E80(__int64 a1, int **a2)
{
  int *v2; // rax
  _BYTE *v3; // rcx
  _BYTE *v4; // rdx
  _BYTE *v5; // r8
  _BYTE *v6; // rdi
  int *v8; // rax
  int v9; // edx

  v2 = *a2;
  if ( (char *)a2[1] - (char *)*a2 != 192 )
    return 1;
  v3 = (_BYTE *)*((_QWORD *)v2 + 1);
  v4 = (_BYTE *)*((_QWORD *)v2 + 13);
  v5 = (_BYTE *)*((_QWORD *)v2 + 15);
  v6 = (_BYTE *)*((_QWORD *)v2 + 3);
  if ( v3 == v4 )
    return v6 != v5 && (v3 != v6 || v3 != v5);
  if ( v4 == v6 && v3 == v5 )
    return 0;
  if ( v6 != v5 || *v2 != v2[24] || *v6 > 0x15u )
    return 1;
  if ( !sub_AC30F0((__int64)v6) )
    return 1;
  v8 = *a2;
  v9 = **a2;
  if ( v9 != 17 )
  {
    if ( v9 == 22 )
      return *((_QWORD *)v8 + 5) != *((_QWORD *)v8 + 18);
    return 1;
  }
  return *((_QWORD *)v8 + 4) != *((_QWORD *)v8 + 18);
}
