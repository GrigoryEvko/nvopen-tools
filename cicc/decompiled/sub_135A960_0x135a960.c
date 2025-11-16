// Function: sub_135A960
// Address: 0x135a960
//
__int64 __fastcall sub_135A960(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rdi
  __int64 result; // rax
  __int64 v6; // [rsp+8h] [rbp-18h] BYREF

  v6 = a2;
  v3 = *(_QWORD *)(a1 + 48);
  if ( v3 == *(_QWORD *)(a1 + 40) )
  {
    *(_DWORD *)(a1 + 64) = (*(_DWORD *)(a1 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(a1 + 64) & 0xF8000000;
    if ( v3 != *(_QWORD *)(a1 + 56) )
      goto LABEL_3;
LABEL_13:
    sub_135A6D0((char **)(a1 + 40), (char *)v3, &v6);
    v4 = v6;
    goto LABEL_9;
  }
  if ( v3 == *(_QWORD *)(a1 + 56) )
    goto LABEL_13;
LABEL_3:
  v4 = v6;
  if ( v3 )
  {
    *(_QWORD *)v3 = 4;
    *(_QWORD *)(v3 + 8) = 0;
    *(_QWORD *)(v3 + 16) = v4;
    if ( v4 != 0 && v4 != -8 && v4 != -16 )
      sub_164C220(v3);
    v3 = *(_QWORD *)(a1 + 48);
    v4 = v6;
  }
  *(_QWORD *)(a1 + 48) = v3 + 24;
LABEL_9:
  result = sub_15F3040(v4);
  if ( (_BYTE)result )
  {
    *(_BYTE *)(a1 + 67) |= 0x70u;
  }
  else
  {
    *(_BYTE *)(a1 + 67) |= 0x40u;
    *(_BYTE *)(a1 + 67) |= 0x10u;
  }
  return result;
}
