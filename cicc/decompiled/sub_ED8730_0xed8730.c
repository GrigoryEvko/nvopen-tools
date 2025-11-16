// Function: sub_ED8730
// Address: 0xed8730
//
__int64 *__fastcall sub_ED8730(__int64 *a1, int *a2, _BYTE *a3)
{
  bool v3; // zf
  int v4; // r13d
  __int64 v5; // rax
  __int64 v6; // rbx
  void *v8; // [rsp+0h] [rbp-50h] BYREF
  __int16 v9; // [rsp+20h] [rbp-30h]

  v3 = *a3 == 0;
  v4 = *a2;
  v9 = 257;
  if ( !v3 )
  {
    v8 = a3;
    LOBYTE(v9) = 3;
  }
  v5 = sub_22077B0(48);
  v6 = v5;
  if ( v5 )
  {
    *(_DWORD *)(v5 + 8) = v4;
    *(_QWORD *)v5 = &unk_49E4BC8;
    sub_CA0F50((__int64 *)(v5 + 16), &v8);
  }
  *a1 = v6 | 1;
  return a1;
}
