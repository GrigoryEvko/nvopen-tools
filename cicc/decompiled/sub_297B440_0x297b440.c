// Function: sub_297B440
// Address: 0x297b440
//
_QWORD *sub_297B440()
{
  _QWORD *v0; // rax
  _QWORD *v1; // r12
  __int64 v2; // rdi
  char v3; // al

  v0 = (_QWORD *)sub_22077B0(0xC0u);
  v1 = v0;
  if ( v0 )
  {
    v0[1] = 0;
    v0[2] = &unk_500708C;
    v2 = (__int64)(v0 + 22);
    v0[7] = v0 + 13;
    v0[14] = v0 + 20;
    *v0 = off_4A22290;
    v3 = qword_5007128;
    *((_DWORD *)v1 + 6) = 2;
    v1[4] = 0;
    v1[5] = 0;
    v1[6] = 0;
    v1[8] = 1;
    v1[9] = 0;
    v1[10] = 0;
    v1[12] = 0;
    v1[13] = 0;
    v1[15] = 1;
    v1[16] = 0;
    v1[17] = 0;
    v1[19] = 0;
    v1[20] = 0;
    *((_BYTE *)v1 + 168) = 0;
    *((_BYTE *)v1 + 169) = v3;
    *((_DWORD *)v1 + 22) = 1065353216;
    *((_DWORD *)v1 + 36) = 1065353216;
    sub_297B2F0(v2, 0);
  }
  return v1;
}
