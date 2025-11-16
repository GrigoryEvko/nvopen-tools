// Function: sub_1E6EEE0
// Address: 0x1e6eee0
//
_QWORD *sub_1E6EEE0()
{
  _QWORD *v0; // rax
  _QWORD *v1; // r12
  _QWORD *v2; // rax
  _QWORD *v3; // rax
  _QWORD *v4; // rax
  __int64 v5; // rax

  v0 = (_QWORD *)sub_22077B0(296);
  v1 = v0;
  if ( v0 )
  {
    sub_1E6ED70(v0);
    v1[9] = 0;
    v1[10] = &unk_4FC7874;
    v1[18] = v1 + 16;
    v1[19] = v1 + 16;
    v1[24] = v1 + 22;
    v1[25] = v1 + 22;
    *((_DWORD *)v1 + 22) = 3;
    v1[12] = 0;
    v1[13] = 0;
    v1[14] = 0;
    *((_DWORD *)v1 + 32) = 0;
    v1[17] = 0;
    v1[20] = 0;
    *((_DWORD *)v1 + 44) = 0;
    v1[23] = 0;
    v1[26] = 0;
    *((_BYTE *)v1 + 216) = 0;
    v1[8] = &unk_49FB790;
    v1[28] = 0;
    v1[29] = 0;
    *((_DWORD *)v1 + 60) = 8;
    v2 = (_QWORD *)malloc(8u);
    if ( !v2 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v2 = 0;
    }
    *v2 = 0;
    v1[28] = v2;
    v1[29] = 1;
    v1[31] = 0;
    v1[32] = 0;
    *((_DWORD *)v1 + 66) = 8;
    v3 = (_QWORD *)malloc(8u);
    if ( !v3 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v3 = 0;
    }
    *v3 = 0;
    v1[31] = v3;
    v1[32] = 1;
    v1[34] = 0;
    v1[35] = 0;
    *((_DWORD *)v1 + 72) = 8;
    v4 = (_QWORD *)malloc(8u);
    if ( !v4 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v4 = 0;
    }
    *v4 = 0;
    v1 += 8;
    v1[26] = v4;
    *(v1 - 8) = off_49FC478;
    v1[27] = 1;
    *v1 = &unk_49FC4B0;
    v5 = sub_163A1D0();
    sub_1E6EDE0(v5);
  }
  return v1;
}
