// Function: sub_C37B40
// Address: 0xc37b40
//
unsigned int *__fastcall sub_C37B40(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 *v3; // r13
  __int64 v4; // rbx
  __int64 v5; // r15
  char v6; // si
  __int64 v7; // r14
  unsigned int *result; // rax
  _QWORD *v9; // rax
  char v10; // al

  v2 = *((unsigned int *)a2 + 2);
  if ( (unsigned int)v2 > 0x40 )
    a2 = (__int64 *)*a2;
  v3 = (__int64 *)*a2;
  v4 = a2[1] & 0xFFFFFFFFFFFFLL;
  v5 = a2[(unsigned int)((unsigned __int64)(v2 + 63) >> 6) - 1];
  sub_C337F0((_QWORD *)a1, (__int64)&unk_3F65780);
  v6 = (8 * (v5 < 0)) | *(_BYTE *)(a1 + 20) & 0xF7;
  v7 = HIWORD(v5) & 0x7FFF;
  *(_BYTE *)(a1 + 20) = v6;
  if ( v3 || v4 )
  {
    if ( v7 != 0x7FFF )
    {
      *(_BYTE *)(a1 + 20) = *(_BYTE *)(a1 + 20) & 0xF8 | 2;
      *(_DWORD *)(a1 + 16) = v7 - 0x3FFF;
      result = (unsigned int *)sub_C33900(a1);
      *(_QWORD *)result = v3;
      *((_QWORD *)result + 1) = v4;
      if ( !v7 )
      {
        *(_DWORD *)(a1 + 16) = -16382;
        return result;
      }
      goto LABEL_12;
    }
    v10 = *(_BYTE *)(a1 + 20);
    *(_DWORD *)(a1 + 16) = 0x4000;
    *(_BYTE *)(a1 + 20) = v10 & 0xF8 | 1;
    result = (unsigned int *)sub_C33900(a1);
    *(_QWORD *)result = v3;
    *((_QWORD *)result + 1) = v4;
  }
  else
  {
    if ( v7 != 0x7FFF )
    {
      if ( !v7 )
        return (unsigned int *)sub_C37310(a1, (v6 & 8) != 0);
      *(_DWORD *)(a1 + 16) = v7 - 0x3FFF;
      *(_BYTE *)(a1 + 20) = v6 & 0xF8 | 2;
      v9 = (_QWORD *)sub_C33900(a1);
      *v9 = 0;
      v9[1] = 0;
LABEL_12:
      result = (unsigned int *)sub_C33900(a1);
      *((_QWORD *)result + 1) |= 0x1000000000000uLL;
      return result;
    }
    return sub_C36EF0((_DWORD **)a1, (v6 & 8) != 0);
  }
  return result;
}
