// Function: sub_7E3260
// Address: 0x7e3260
//
char *__fastcall sub_7E3260(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  const char *v4; // r12
  size_t v5; // rax
  char *v6; // rax
  char *result; // rax

  v4 = (const char *)sub_810D50(a2, a3, a4);
  v5 = strlen(v4);
  v6 = (char *)sub_7E1510(v5 + 1);
  result = strcpy(v6, v4);
  *(_BYTE *)(a1 + 89) |= 8u;
  *(_QWORD *)(a1 + 8) = result;
  return result;
}
