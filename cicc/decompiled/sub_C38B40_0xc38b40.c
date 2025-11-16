// Function: sub_C38B40
// Address: 0xc38b40
//
unsigned int *__fastcall sub_C38B40(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _QWORD *v3; // rdx
  __int64 v4; // rbx
  unsigned __int64 v5; // r13
  char v6; // si
  __int64 v7; // r14
  unsigned int *result; // rax

  v2 = *(unsigned int *)(a2 + 8);
  v3 = *(_QWORD **)a2;
  if ( (unsigned int)v2 <= 0x40 )
  {
    v4 = *(_QWORD *)a2 & 0xFLL;
    v3 = (_QWORD *)a2;
  }
  else
  {
    v4 = *v3 & 0xFLL;
  }
  v5 = v3[(unsigned int)((unsigned __int64)(v2 + 63) >> 6) - 1];
  sub_C337F0((_QWORD *)a1, (__int64)&unk_3F656A0);
  v6 = (8 * ((unsigned __int8)v5 >> 7)) | *(_BYTE *)(a1 + 20) & 0xF7;
  v7 = (v5 >> 4) & 7;
  *(_BYTE *)(a1 + 20) = v6;
  if ( v4 )
  {
    if ( v7 == 7 )
    {
      *(_DWORD *)(a1 + 16) = 4;
      *(_BYTE *)(a1 + 20) = v6 & 0xF8 | 1;
      result = (unsigned int *)sub_C33900(a1);
      *(_QWORD *)result = v4;
    }
    else
    {
      *(_DWORD *)(a1 + 16) = v7 - 3;
      *(_BYTE *)(a1 + 20) = v6 & 0xF8 | 2;
      result = (unsigned int *)sub_C33900(a1);
      *(_QWORD *)result = v4;
      if ( v7 )
        goto LABEL_7;
      *(_DWORD *)(a1 + 16) = -2;
    }
  }
  else if ( v7 == 7 )
  {
    return sub_C36EF0((_DWORD **)a1, (v6 & 8) != 0);
  }
  else
  {
    if ( v7 )
    {
      *(_DWORD *)(a1 + 16) = v7 - 3;
      *(_BYTE *)(a1 + 20) = v6 & 0xF8 | 2;
      *(_QWORD *)sub_C33900(a1) = 0;
LABEL_7:
      result = (unsigned int *)sub_C33900(a1);
      *(_QWORD *)result |= 0x10uLL;
      return result;
    }
    return (unsigned int *)sub_C37310(a1, (v6 & 8) != 0);
  }
  return result;
}
