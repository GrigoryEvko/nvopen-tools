// Function: sub_ED0840
// Address: 0xed0840
//
__int64 *__fastcall sub_ED0840(__int64 *a1, int a2, _BYTE *a3)
{
  bool v3; // zf
  __int64 v4; // rax
  __int64 v5; // rbx
  void *v7; // [rsp+0h] [rbp-50h] BYREF
  __int16 v8; // [rsp+20h] [rbp-30h]

  v3 = *a3 == 0;
  v8 = 257;
  if ( !v3 )
  {
    v7 = a3;
    LOBYTE(v8) = 3;
  }
  v4 = sub_22077B0(48);
  v5 = v4;
  if ( v4 )
  {
    *(_DWORD *)(v4 + 8) = a2;
    *(_QWORD *)v4 = &unk_49E4BC8;
    sub_CA0F50((__int64 *)(v4 + 16), &v7);
  }
  *a1 = v5 | 1;
  return a1;
}
