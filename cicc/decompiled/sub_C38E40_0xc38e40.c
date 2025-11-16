// Function: sub_C38E40
// Address: 0xc38e40
//
_QWORD *__fastcall sub_C38E40(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _QWORD *v3; // rdx
  __int64 v4; // r12
  unsigned __int64 v5; // r13
  __int64 v6; // rbx
  char v7; // si
  _QWORD *result; // rax

  v2 = *(unsigned int *)(a2 + 8);
  v3 = *(_QWORD **)a2;
  if ( (unsigned int)v2 <= 0x40 )
  {
    v4 = *(_QWORD *)a2 & 3LL;
    v3 = (_QWORD *)a2;
  }
  else
  {
    v4 = *v3 & 3LL;
  }
  v5 = v3[(unsigned int)((unsigned __int64)(v2 + 63) >> 6) - 1];
  sub_C337F0((_QWORD *)a1, (__int64)&unk_3F65640);
  v6 = (v5 >> 2) & 7;
  v7 = (8 * ((v5 & 0x20) != 0)) | *(_BYTE *)(a1 + 20) & 0xF7;
  *(_BYTE *)(a1 + 20) = v7;
  if ( v4 )
  {
    *(_DWORD *)(a1 + 16) = v6 - 3;
    *(_BYTE *)(a1 + 20) = v7 & 0xF8 | 2;
    result = (_QWORD *)sub_C33900(a1);
    *result = v4;
    if ( v6 )
      goto LABEL_6;
    *(_DWORD *)(a1 + 16) = -2;
  }
  else
  {
    if ( v6 )
    {
      *(_DWORD *)(a1 + 16) = v6 - 3;
      *(_BYTE *)(a1 + 20) = v7 & 0xF8 | 2;
      *(_QWORD *)sub_C33900(a1) = 0;
LABEL_6:
      result = (_QWORD *)sub_C33900(a1);
      *result |= 4uLL;
      return result;
    }
    return (_QWORD *)sub_C37310(a1, (v7 & 8) != 0);
  }
  return result;
}
