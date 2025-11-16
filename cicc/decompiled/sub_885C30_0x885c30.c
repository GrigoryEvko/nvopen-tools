// Function: sub_885C30
// Address: 0x885c30
//
__int64 __fastcall sub_885C30(__int16 a1, char *a2)
{
  size_t v2; // rax
  _QWORD *v3; // rax
  __int64 v4; // rdx
  __int64 result; // rax

  v2 = strlen(a2);
  v3 = sub_885B80(a2, v2, 0, -1);
  v4 = *v3;
  *((_WORD *)v3 + 44) = a1;
  *(_BYTE *)(v4 + 73) |= 0x20u;
  result = *v3;
  *(_BYTE *)(result + 74) = 8;
  return result;
}
