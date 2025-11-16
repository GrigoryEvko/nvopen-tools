// Function: sub_2C1BAD0
// Address: 0x2c1bad0
//
__int64 __fastcall sub_2C1BAD0(__int64 a1)
{
  _QWORD *v1; // r13
  unsigned __int8 *v2; // r14
  _QWORD *v3; // r15
  __int64 v4; // rax
  __int64 v5; // r9
  __int64 v6; // r12
  int v7; // eax

  v1 = *(_QWORD **)(a1 + 48);
  v2 = *(unsigned __int8 **)(a1 + 136);
  v3 = &v1[*(unsigned int *)(a1 + 56)];
  v4 = sub_22077B0(0xA8u);
  v6 = v4;
  if ( v4 )
  {
    sub_2ABDBC0(v4, 23, v1, v3, v2, v5);
    *(_QWORD *)v6 = &unk_4A23EC8;
    *(_QWORD *)(v6 + 96) = &unk_4A23F38;
    v7 = *v2;
    *(_QWORD *)(v6 + 40) = &unk_4A23F00;
    *(_DWORD *)(v6 + 160) = v7 - 29;
  }
  *(_BYTE *)(v6 + 152) = *(_BYTE *)(a1 + 152);
  *(_DWORD *)(v6 + 156) = *(_DWORD *)(a1 + 156);
  return v6;
}
