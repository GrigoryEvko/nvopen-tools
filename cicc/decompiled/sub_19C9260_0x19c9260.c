// Function: sub_19C9260
// Address: 0x19c9260
//
_QWORD *sub_19C9260()
{
  _QWORD *v0; // rax
  _QWORD *v1; // r12
  int v2; // xmm0_4
  int v3; // eax
  __int64 v4; // rax

  v0 = (_QWORD *)sub_22077B0(240);
  v1 = v0;
  if ( v0 )
  {
    v0[1] = 0;
    v2 = dword_4FB38E0;
    v0[2] = &unk_4FB374C;
    v0[10] = v0 + 8;
    v0[11] = v0 + 8;
    v0[16] = v0 + 14;
    v0[17] = v0 + 14;
    *v0 = off_49F4790;
    v3 = dword_4FB3800;
    *((_DWORD *)v1 + 6) = 2;
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
    v1[20] = 0;
    v1[21] = 0;
    v1[22] = 0;
    v1[23] = 0;
    v1[24] = 0;
    v1[25] = 0;
    *((_DWORD *)v1 + 52) = v3;
    v1[27] = 0;
    *((_BYTE *)v1 + 224) = 1;
    *((_DWORD *)v1 + 53) = v2;
    v4 = sub_163A1D0();
    sub_19C9140(v4);
  }
  return v1;
}
