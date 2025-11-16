// Function: sub_254E600
// Address: 0x254e600
//
__int64 __fastcall sub_254E600(__int64 *a1, __int64 a2)
{
  __int64 *v2; // r15
  unsigned int v3; // r14d
  __int64 v6; // rax
  void (__fastcall *v7)(__int64 *, __int64, __int64, _BYTE **); // r14
  unsigned __int64 v8; // rdi
  __int64 v9; // rax
  _BYTE *v10; // [rsp+0h] [rbp-60h] BYREF
  __int64 v11; // [rsp+8h] [rbp-58h]
  _BYTE v12[80]; // [rsp+10h] [rbp-50h] BYREF

  v2 = a1 + 9;
  v3 = 1;
  if ( (unsigned int)*(unsigned __int8 *)sub_250D070(a1 + 9) - 12 > 1 )
  {
    v11 = 0x400000000LL;
    v6 = *a1;
    v10 = v12;
    v7 = *(void (__fastcall **)(__int64 *, __int64, __int64, _BYTE **))(v6 + 112);
    v8 = a1[9] & 0xFFFFFFFFFFFFFFFCLL;
    if ( (a1[9] & 3) == 3 )
      v8 = *(_QWORD *)(v8 + 24);
    v9 = sub_BD5C60(v8);
    v7(a1, a2, v9, &v10);
    v3 = 1;
    if ( (_DWORD)v11 )
      v3 = sub_2516380(a2, v2, (__int64)v10, (unsigned int)v11, 0);
    if ( v10 != v12 )
      _libc_free((unsigned __int64)v10);
  }
  return v3;
}
