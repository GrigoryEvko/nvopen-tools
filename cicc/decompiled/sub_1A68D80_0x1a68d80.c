// Function: sub_1A68D80
// Address: 0x1a68d80
//
_QWORD *sub_1A68D80()
{
  _QWORD *v0; // rax
  _QWORD *v1; // r12
  __int64 v2; // rdi
  char v3; // al

  v0 = (_QWORD *)sub_22077B0(176);
  v1 = v0;
  if ( v0 )
  {
    v0[1] = 0;
    v2 = (__int64)(v0 + 20);
    v0[2] = &unk_4FB4ACC;
    v0[10] = v0 + 8;
    v0[11] = v0 + 8;
    v0[16] = v0 + 14;
    v0[17] = v0 + 14;
    *v0 = off_49F5678;
    v3 = byte_4FB4B80;
    *((_DWORD *)v1 + 6) = 3;
    v1[4] = 0;
    v1[5] = 0;
    v1[6] = 0;
    *((_DWORD *)v1 + 16) = 0;
    v1[9] = 0;
    v1[12] = 0;
    *((_DWORD *)v1 + 28) = 0;
    v1[15] = 0;
    v1[18] = 0;
    *((_BYTE *)v1 + 152) = 0;
    *((_BYTE *)v1 + 153) = v3;
    sub_1A68C70(v2, 0);
  }
  return v1;
}
