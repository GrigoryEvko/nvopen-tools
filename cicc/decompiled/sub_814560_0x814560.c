// Function: sub_814560
// Address: 0x814560
//
_BYTE *__fastcall sub_814560(__int64 a1, __int64 a2)
{
  _BYTE *result; // rax
  const char *v3; // r13
  size_t v4; // rax
  char *v5; // rax
  char v6; // al
  _DWORD v7[9]; // [rsp+Ch] [rbp-24h] BYREF

  if ( (*(_BYTE *)(a2 + 89) & 8) != 0 || (result = (_BYTE *)sub_80A070(a2, v7), (_DWORD)result) )
  {
    sub_814390(a2, 0);
    v3 = *(const char **)(a2 + 8);
    v4 = strlen(v3);
    v5 = (char *)sub_7E1510(v4 + 1);
    *(_QWORD *)(a1 + 8) = strcpy(v5, v3);
    v6 = *(_BYTE *)(a1 + 89) | 8;
    *(_BYTE *)(a1 + 89) = v6;
    *(_QWORD *)(a1 + 184) = *(_QWORD *)(a2 + 184);
    *(_BYTE *)(a1 + 89) = *(_BYTE *)(a2 + 89) & 0x10 | v6 & 0xEF;
    return sub_80D2C0(a1);
  }
  return result;
}
