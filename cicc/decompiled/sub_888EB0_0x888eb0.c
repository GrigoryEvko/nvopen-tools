// Function: sub_888EB0
// Address: 0x888eb0
//
__int64 __fastcall sub_888EB0(char *s, __int64 a2)
{
  _QWORD *v2; // rax
  __int64 v3; // r12
  size_t v4; // rax
  __int64 *v5; // rax

  v2 = sub_7259C0(12);
  *((_BYTE *)v2 + 185) |= 0x80u;
  v3 = (__int64)v2;
  v2[20] = a2;
  sub_7365B0((__int64)v2, 0);
  v4 = strlen(s);
  v5 = sub_885B80(s, v4, 3u, 0);
  v5[11] = v3;
  sub_877D80(v3, v5);
  return v3;
}
