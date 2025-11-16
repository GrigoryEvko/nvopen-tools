// Function: sub_C373D0
// Address: 0xc373d0
//
unsigned int *__fastcall sub_C373D0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rbx
  __int64 v3; // r13
  __int64 v4; // r14
  int v5; // ebx
  unsigned int *result; // rax
  char v7; // al

  if ( *(_DWORD *)(a2 + 8) > 0x40u )
    a2 = *(_QWORD *)a2;
  v2 = *(_QWORD *)(a2 + 8);
  v3 = *(_QWORD *)a2;
  sub_C337F0((_QWORD *)a1, (__int64)&unk_3F655E0);
  v4 = v2 & 0x7FFF;
  v5 = *(_BYTE *)(a1 + 20) & 0xF7 | (8 * ((v2 >> 15) & 1));
  *(_BYTE *)(a1 + 20) = v5;
  if ( !(v4 | v3) )
    return (unsigned int *)sub_C37310(a1, (v5 & 8) != 0);
  if ( v3 == 0x8000000000000000LL && v4 == 0x7FFF )
    return sub_C36EF0((_DWORD **)a1, (v5 & 8) != 0);
  if ( v3 != 0x8000000000000000LL && v4 == 0x7FFF )
    goto LABEL_12;
  if ( v4 != 0x7FFF && v4 )
  {
    if ( v3 < 0 )
    {
      v7 = *(_BYTE *)(a1 + 20);
      *(_DWORD *)(a1 + 16) = v4 - 0x3FFF;
      *(_BYTE *)(a1 + 20) = v7 & 0xF8 | 2;
      goto LABEL_13;
    }
LABEL_12:
    *(_BYTE *)(a1 + 20) = *(_BYTE *)(a1 + 20) & 0xF8 | 1;
    *(_DWORD *)(a1 + 16) = sub_C36030((unsigned int **)a1);
LABEL_13:
    *(_QWORD *)sub_C33900(a1) = v3;
    result = (unsigned int *)sub_C33900(a1);
    *((_QWORD *)result + 1) = 0;
    return result;
  }
  *(_BYTE *)(a1 + 20) = *(_BYTE *)(a1 + 20) & 0xF8 | 2;
  *(_DWORD *)(a1 + 16) = v4 - 0x3FFF;
  *(_QWORD *)sub_C33900(a1) = v3;
  result = (unsigned int *)sub_C33900(a1);
  *((_QWORD *)result + 1) = 0;
  if ( !v4 )
    *(_DWORD *)(a1 + 16) = -16382;
  return result;
}
