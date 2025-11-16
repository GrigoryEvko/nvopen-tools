// Function: sub_1097E90
// Address: 0x1097e90
//
bool __fastcall sub_1097E90(__int64 a1, const char *a2)
{
  const char *v2; // r13
  size_t v3; // rax

  v2 = *(const char **)(*(_QWORD *)(a1 + 144) + 40LL);
  v3 = strlen(v2);
  return strncmp(a2, v2, v3) == 0;
}
