// Function: sub_2559CE0
// Address: 0x2559ce0
//
__int64 __fastcall sub_2559CE0(__int64 *a1, __int64 a2)
{
  unsigned int v2; // r15d
  __int64 *v3; // r14
  __int64 v6; // rax
  __int64 (__fastcall *v7)(__int64, __int64, __int64 *, __int64); // r15
  unsigned __int64 v8; // rdi
  __int64 *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // [rsp+8h] [rbp-68h] BYREF
  _BYTE *v15; // [rsp+10h] [rbp-60h] BYREF
  __int64 v16; // [rsp+18h] [rbp-58h]
  _BYTE v17[80]; // [rsp+20h] [rbp-50h] BYREF

  v2 = 1;
  v3 = a1 + 9;
  if ( (unsigned int)*(unsigned __int8 *)sub_250D070(a1 + 9) - 12 > 1 )
  {
    v16 = 0x400000000LL;
    v6 = *a1;
    v15 = v17;
    v7 = *(__int64 (__fastcall **)(__int64, __int64, __int64 *, __int64))(v6 + 112);
    v8 = a1[9] & 0xFFFFFFFFFFFFFFFCLL;
    if ( (a1[9] & 3) == 3 )
      v8 = *(_QWORD *)(v8 + 24);
    v9 = (__int64 *)sub_BD5C60(v8);
    if ( v7 == sub_2547240 )
    {
      v14 = sub_A778C0(v9, 19, 0);
      sub_25594F0((__int64)&v15, &v14, v10, v11, v12, v13);
    }
    else
    {
      v7((__int64)a1, a2, v9, (__int64)&v15);
    }
    v2 = 1;
    if ( (_DWORD)v16 )
      v2 = sub_2516380(a2, v3, (__int64)v15, (unsigned int)v16, 0);
    if ( v15 != v17 )
      _libc_free((unsigned __int64)v15);
  }
  return v2;
}
