// Function: sub_26725F0
// Address: 0x26725f0
//
__int64 __fastcall sub_26725F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r12d
  int v8; // eax
  __int64 v9; // rcx
  int v10; // edx
  unsigned int v11; // eax
  unsigned __int64 v12; // rbx
  __int64 v13; // rdi
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  int v17; // [rsp+0h] [rbp-90h]
  _BYTE v18[8]; // [rsp+8h] [rbp-88h] BYREF
  unsigned __int64 v19; // [rsp+10h] [rbp-80h]
  char v20; // [rsp+24h] [rbp-6Ch]
  _BYTE v21[16]; // [rsp+28h] [rbp-68h] BYREF
  _BYTE v22[8]; // [rsp+38h] [rbp-58h] BYREF
  unsigned __int64 v23; // [rsp+40h] [rbp-50h]
  char v24; // [rsp+54h] [rbp-3Ch]
  _BYTE v25[56]; // [rsp+58h] [rbp-38h] BYREF

  v6 = *(unsigned __int8 *)(a1 + 97);
  if ( !(_BYTE)v6 )
    return v6;
  v8 = *(_DWORD *)(a1 + 248);
  v9 = *(_QWORD *)(a1 + 232);
  if ( !v8 )
    return v6;
  v10 = v8 - 1;
  v11 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v12 = v9 + ((unsigned __int64)v11 << 7);
  v13 = *(_QWORD *)v12;
  if ( a2 != *(_QWORD *)v12 )
  {
    a5 = 1;
    while ( v13 != -4096 )
    {
      a6 = (unsigned int)(a5 + 1);
      v11 = v10 & (a5 + v11);
      v12 = v9 + ((unsigned __int64)v11 << 7);
      v13 = *(_QWORD *)v12;
      if ( a2 == *(_QWORD *)v12 )
        goto LABEL_5;
      a5 = (unsigned int)a6;
    }
    return v6;
  }
LABEL_5:
  v17 = *(_DWORD *)(v12 + 8);
  sub_C8CD80((__int64)v18, (__int64)v21, v12 + 16, v9, a5, a6);
  sub_C8CD80((__int64)v22, (__int64)v25, v12 + 64, v14, v15, v16);
  v6 = (unsigned __int8)v17;
  if ( !v24 )
    _libc_free(v23);
  if ( v20 )
    return v6;
  _libc_free(v19);
  return (unsigned __int8)v17;
}
