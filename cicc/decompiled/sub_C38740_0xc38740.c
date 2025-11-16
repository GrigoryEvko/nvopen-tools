// Function: sub_C38740
// Address: 0xc38740
//
_QWORD *__fastcall sub_C38740(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _QWORD *v3; // rdx
  __int64 v4; // rbx
  unsigned __int64 v5; // r13
  char v6; // si
  __int64 v7; // r14
  _QWORD *result; // rax

  v2 = *(unsigned int *)(a2 + 8);
  v3 = *(_QWORD **)a2;
  if ( (unsigned int)v2 <= 0x40 )
  {
    v4 = *(_QWORD *)a2 & 7LL;
    v3 = (_QWORD *)a2;
  }
  else
  {
    v4 = *v3 & 7LL;
  }
  v5 = v3[(unsigned int)((unsigned __int64)(v2 + 63) >> 6) - 1];
  sub_C337F0((_QWORD *)a1, (__int64)&unk_3F65700);
  v6 = (8 * ((unsigned __int8)v5 >> 7)) | *(_BYTE *)(a1 + 20) & 0xF7;
  v7 = (v5 >> 3) & 0xF;
  *(_BYTE *)(a1 + 20) = v6;
  if ( v4 )
  {
    if ( v4 == 7 && v7 == 15 )
    {
      *(_DWORD *)(a1 + 16) = 8;
      *(_BYTE *)(a1 + 20) = v6 & 0xF8 | 1;
      result = (_QWORD *)sub_C33900(a1);
      *result = 7;
      return result;
    }
    *(_BYTE *)(a1 + 20) = *(_BYTE *)(a1 + 20) & 0xF8 | 2;
    *(_DWORD *)(a1 + 16) = v7 - 7;
    result = (_QWORD *)sub_C33900(a1);
    *result = v4;
    if ( !v7 )
    {
      *(_DWORD *)(a1 + 16) = -6;
      return result;
    }
  }
  else
  {
    if ( !v7 )
      return (_QWORD *)sub_C37310(a1, (v6 & 8) != 0);
    *(_DWORD *)(a1 + 16) = v7 - 7;
    *(_BYTE *)(a1 + 20) = v6 & 0xF8 | 2;
    *(_QWORD *)sub_C33900(a1) = 0;
  }
  result = (_QWORD *)sub_C33900(a1);
  *result |= 8uLL;
  return result;
}
