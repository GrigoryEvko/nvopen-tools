// Function: sub_2E25970
// Address: 0x2e25970
//
void *__fastcall sub_2E25970(__int64 a1, _BYTE *a2)
{
  void *v2; // r8
  _BYTE *v3; // rsi
  _BYTE *v4; // rdx
  void *v5; // rax

  v2 = a2;
  v3 = a2 + 8;
  v4 = *(_BYTE **)(a1 + 8);
  if ( v3 != v4 )
  {
    v5 = memmove(v2, v3, v4 - v3);
    v4 = *(_BYTE **)(a1 + 8);
    v2 = v5;
  }
  *(_QWORD *)(a1 + 8) = v4 - 8;
  return v2;
}
