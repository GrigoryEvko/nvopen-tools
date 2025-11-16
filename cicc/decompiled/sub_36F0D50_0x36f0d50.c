// Function: sub_36F0D50
// Address: 0x36f0d50
//
_DWORD *__fastcall sub_36F0D50(__int64 a1, _QWORD *a2)
{
  _BYTE *v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  _DWORD *result; // rax
  unsigned __int8 v24[33]; // [rsp+Fh] [rbp-21h] BYREF

  memset((void *)a1, 0, 0xA0u);
  *(_BYTE *)(a1 + 8) = 1;
  v2 = (_BYTE *)(a1 + 16);
  do
  {
    if ( v2 )
      *v2 = -1;
    v2 += 8;
  }
  while ( (_BYTE *)(a1 + 80) != v2 );
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)(a1 + 88) = 0x800000000LL;
  v24[0] = sub_B6F810(a2, (__int64)"singlethread", 12);
  *(_DWORD *)sub_36F0A50(a1, v24, v3, v4, v5, v6) = 0;
  v24[0] = sub_B6F810(a2, (__int64)byte_3F871B3, 0);
  *(_DWORD *)sub_36F0A50(a1, v24, v7, v8, v9, v10) = 4;
  v24[0] = sub_B6F810(a2, (__int64)"block", 5);
  *(_DWORD *)sub_36F0A50(a1, v24, v11, v12, v13, v14) = 1;
  v24[0] = sub_B6F810(a2, (__int64)"cluster", 7);
  *(_DWORD *)sub_36F0A50(a1, v24, v15, v16, v17, v18) = 2;
  v24[0] = sub_B6F810(a2, (__int64)"device", 6);
  result = (_DWORD *)sub_36F0A50(a1, v24, v19, v20, v21, v22);
  *result = 3;
  return result;
}
