// Function: sub_305FCF0
// Address: 0x305fcf0
//
_QWORD *__fastcall sub_305FCF0(_QWORD *a1)
{
  _DWORD *v1; // rdx

  v1 = (_DWORD *)sub_22077B0(0x3D8u);
  if ( v1 )
  {
    memset(v1, 0, 0x3D8u);
    v1[234] = 65792;
    *(_QWORD *)v1 = &unk_4A2F0A0;
  }
  *a1 = v1;
  return a1;
}
