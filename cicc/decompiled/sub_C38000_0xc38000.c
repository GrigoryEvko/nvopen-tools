// Function: sub_C38000
// Address: 0xc38000
//
unsigned int *__fastcall sub_C38000(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _QWORD *v3; // rdx
  __int64 v4; // rbx
  unsigned __int64 v5; // r13
  __int64 v6; // r14
  char v7; // si
  unsigned int *result; // rax

  v2 = *(unsigned int *)(a2 + 8);
  v3 = *(_QWORD **)a2;
  if ( (unsigned int)v2 <= 0x40 )
  {
    v4 = *(_QWORD *)a2 & 0x7FLL;
    v3 = (_QWORD *)a2;
  }
  else
  {
    v4 = *v3 & 0x7FLL;
  }
  v5 = v3[(unsigned int)((unsigned __int64)(v2 + 63) >> 6) - 1];
  sub_C337F0((_QWORD *)a1, (__int64)&unk_3F657E0);
  v6 = (unsigned __int8)(v5 >> 7);
  v7 = (8 * ((v5 & 0x8000) != 0)) | *(_BYTE *)(a1 + 20) & 0xF7;
  *(_BYTE *)(a1 + 20) = v7;
  if ( v4 )
  {
    if ( v6 == 255 )
    {
      *(_DWORD *)(a1 + 16) = 128;
      *(_BYTE *)(a1 + 20) = v7 & 0xF8 | 1;
      result = (unsigned int *)sub_C33900(a1);
      *(_QWORD *)result = v4;
    }
    else
    {
      *(_DWORD *)(a1 + 16) = v6 - 127;
      *(_BYTE *)(a1 + 20) = v7 & 0xF8 | 2;
      result = (unsigned int *)sub_C33900(a1);
      *(_QWORD *)result = v4;
      if ( (unsigned __int8)(v5 >> 7) )
        goto LABEL_7;
      *(_DWORD *)(a1 + 16) = -126;
    }
  }
  else if ( v6 == 255 )
  {
    return sub_C36EF0((_DWORD **)a1, (v7 & 8) != 0);
  }
  else
  {
    if ( (unsigned __int8)(v5 >> 7) )
    {
      *(_DWORD *)(a1 + 16) = v6 - 127;
      *(_BYTE *)(a1 + 20) = v7 & 0xF8 | 2;
      *(_QWORD *)sub_C33900(a1) = 0;
LABEL_7:
      result = (unsigned int *)sub_C33900(a1);
      *(_QWORD *)result |= 0x80uLL;
      return result;
    }
    return (unsigned int *)sub_C37310(a1, (v7 & 8) != 0);
  }
  return result;
}
