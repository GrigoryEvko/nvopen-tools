// Function: sub_C39060
// Address: 0xc39060
//
_QWORD *__fastcall sub_C39060(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _QWORD *v3; // rdx
  __int64 v4; // r13
  unsigned __int64 v5; // r12
  __int64 v6; // rbx
  char v7; // si
  _QWORD *result; // rax

  v2 = *(unsigned int *)(a2 + 8);
  v3 = *(_QWORD **)a2;
  if ( (unsigned int)v2 <= 0x40 )
  {
    v4 = *(_QWORD *)a2 & 1LL;
    v3 = (_QWORD *)a2;
  }
  else
  {
    v4 = *v3 & 1LL;
  }
  v5 = v3[(unsigned int)((unsigned __int64)(v2 + 63) >> 6) - 1];
  sub_C337F0((_QWORD *)a1, (__int64)&unk_3F65600);
  v6 = (v5 >> 1) & 3;
  v7 = (8 * ((v5 & 8) != 0)) | *(_BYTE *)(a1 + 20) & 0xF7;
  *(_BYTE *)(a1 + 20) = v7;
  if ( v4 )
  {
    *(_DWORD *)(a1 + 16) = v6 - 1;
    *(_BYTE *)(a1 + 20) = v7 & 0xF8 | 2;
    result = (_QWORD *)sub_C33900(a1);
    *result = 1;
    if ( v6 )
      goto LABEL_6;
    *(_DWORD *)(a1 + 16) = 0;
  }
  else
  {
    if ( v6 )
    {
      *(_DWORD *)(a1 + 16) = v6 - 1;
      *(_BYTE *)(a1 + 20) = v7 & 0xF8 | 2;
      *(_QWORD *)sub_C33900(a1) = 0;
LABEL_6:
      result = (_QWORD *)sub_C33900(a1);
      *result |= 2uLL;
      return result;
    }
    return (_QWORD *)sub_C37310(a1, (v7 & 8) != 0);
  }
  return result;
}
