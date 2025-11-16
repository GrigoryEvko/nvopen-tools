// Function: sub_3939A70
// Address: 0x3939a70
//
_QWORD *__fastcall sub_3939A70(_QWORD *a1)
{
  _DWORD *v1; // rdx

  v1 = (_DWORD *)sub_22077B0(0x88u);
  if ( v1 )
  {
    memset(v1, 0, 0x88u);
    v1[11] = 16;
  }
  *a1 = v1;
  return a1;
}
