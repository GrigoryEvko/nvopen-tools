// Function: sub_293BA20
// Address: 0x293ba20
//
_QWORD *__fastcall sub_293BA20(int *a1)
{
  _QWORD *v2; // rax
  _QWORD *v3; // r12
  int v4; // eax
  __int64 v5; // rdi
  void *v7; // rax
  __int64 v8; // rdx
  const void *v9; // rsi

  v2 = (_QWORD *)sub_22077B0(0xD8u);
  v3 = v2;
  if ( !v2 )
    return v3;
  v2[1] = 0;
  v2[2] = &unk_5005714;
  v2[7] = v2 + 13;
  v2[14] = v2 + 20;
  *v2 = off_4A21F48;
  v4 = *a1;
  *((_DWORD *)v3 + 6) = 2;
  *((_DWORD *)v3 + 44) = v4;
  LOWORD(v4) = *((_WORD *)a1 + 2);
  v3[4] = 0;
  v3[5] = 0;
  v3[6] = 0;
  v3[8] = 1;
  v3[9] = 0;
  v3[10] = 0;
  v3[12] = 0;
  v3[13] = 0;
  v3[15] = 1;
  v3[16] = 0;
  v3[17] = 0;
  v3[19] = 0;
  v3[20] = 0;
  *((_BYTE *)v3 + 168) = 0;
  *((_WORD *)v3 + 90) = v4;
  v3[23] = 0;
  v3[24] = 0;
  v3[25] = 0;
  *((_DWORD *)v3 + 52) = 0;
  *((_DWORD *)v3 + 22) = 1065353216;
  *((_DWORD *)v3 + 36) = 1065353216;
  sub_C7D6A0(0, 0, 4);
  v5 = (unsigned int)a1[8];
  *((_DWORD *)v3 + 52) = v5;
  if ( !(_DWORD)v5 )
  {
    v3[24] = 0;
    v3[25] = 0;
    return v3;
  }
  v7 = (void *)sub_C7D670(4 * v5, 4);
  v8 = *((unsigned int *)v3 + 52);
  v9 = (const void *)*((_QWORD *)a1 + 2);
  v3[24] = v7;
  v3[25] = *((_QWORD *)a1 + 3);
  memcpy(v7, v9, 4 * v8);
  return v3;
}
