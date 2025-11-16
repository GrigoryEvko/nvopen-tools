// Function: sub_254E410
// Address: 0x254e410
//
__int64 __fastcall sub_254E410(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rdx
  __int64 v7; // rcx
  unsigned int v8; // r13d
  unsigned __int64 v10; // [rsp+0h] [rbp-40h]
  _BYTE *v11; // [rsp+8h] [rbp-38h] BYREF
  __int64 v12; // [rsp+10h] [rbp-30h]
  _BYTE v13[40]; // [rsp+18h] [rbp-28h] BYREF

  v6 = *(_QWORD *)a2;
  v7 = *(unsigned int *)(a2 + 16);
  v11 = v13;
  v12 = 0;
  v10 = v6;
  if ( (_DWORD)v7 )
  {
    sub_2538240((__int64)&v11, (char **)(a2 + 8), v6, v7, a5, a6);
    if ( (_DWORD)v12 )
      goto LABEL_5;
    v6 = v10;
  }
  if ( v6 )
  {
    v8 = sub_254E2A0(a1[1], *a1, v6);
    goto LABEL_6;
  }
LABEL_5:
  v8 = 0;
LABEL_6:
  if ( v11 != v13 )
    _libc_free((unsigned __int64)v11);
  return v8;
}
