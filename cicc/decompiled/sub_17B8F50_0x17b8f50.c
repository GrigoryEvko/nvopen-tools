// Function: sub_17B8F50
// Address: 0x17b8f50
//
_QWORD *__fastcall sub_17B8F50(unsigned __int64 *a1)
{
  _QWORD *v1; // rax
  _QWORD *v2; // r12
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  __int64 v5; // rax

  v1 = (_QWORD *)sub_22077B0(376);
  v2 = v1;
  if ( v1 )
  {
    v1[1] = 0;
    v3 = *a1;
    v1[2] = &unk_4FA29AC;
    v1[10] = v1 + 8;
    v1[11] = v1 + 8;
    v1[16] = v1 + 14;
    v1[17] = v1 + 14;
    *v1 = off_49F0128;
    v4 = v3;
    v2[20] = v3;
    LOWORD(v3) = *((_WORD *)a1 + 4);
    *((_DWORD *)v2 + 6) = 5;
    *((_WORD *)v2 + 84) = v3;
    v2[22] = v2 + 24;
    v2[4] = 0;
    v2[5] = 0;
    v2[6] = 0;
    *((_DWORD *)v2 + 16) = 0;
    v2[9] = 0;
    v2[12] = 0;
    *((_DWORD *)v2 + 28) = 0;
    v2[15] = 0;
    v2[18] = 0;
    *((_BYTE *)v2 + 152) = 0;
    v2[23] = 0x400000000LL;
    v2[29] = v2 + 31;
    v2[30] = 0x1000000000LL;
    *(_DWORD *)((char *)v2 + 170) = _byteswap_ulong(v4 >> 16);
    *((_BYTE *)v2 + 174) = 0;
    v5 = sub_163A1D0();
    sub_17B8D00(v5);
  }
  return v2;
}
