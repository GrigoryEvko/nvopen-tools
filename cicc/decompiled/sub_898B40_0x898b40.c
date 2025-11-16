// Function: sub_898B40
// Address: 0x898b40
//
__int64 *__fastcall sub_898B40(
        __int64 a1,
        __int64 a2,
        int a3,
        int a4,
        int a5,
        unsigned int a6,
        char a7,
        char a8,
        char a9,
        char a10)
{
  __int64 *v13; // r14
  _QWORD *v14; // rax
  __int64 v15; // rbx
  __int64 v16; // rdx

  v13 = sub_897810(0x13u, a2, a5 == 0, a6);
  v14 = sub_727340();
  *((_BYTE *)v14 + 120) = 8;
  v15 = (__int64)v14;
  sub_877D80((__int64)v14, v13);
  *(_QWORD *)(v15 + 176) = a1;
  if ( !a5 )
    sub_877D70(v15);
  v16 = v13[11];
  *(_QWORD *)(v15 + 168) = v16;
  *(_DWORD *)(v16 + 264) = *(_DWORD *)(v16 + 264) & 0xFFFEFF00 | 0x10009;
  *(_DWORD *)(v15 + 132) = a3;
  *(_DWORD *)(v15 + 128) = a4;
  *(_BYTE *)(v15 + 121) = (4 * (a7 & 1)) | *(_BYTE *)(v15 + 121) & 0xFB;
  *(_QWORD *)(v16 + 104) = v15;
  *(_QWORD *)(v16 + 200) = v13;
  *(_BYTE *)(v16 + 160) = *(_BYTE *)(v16 + 160) & 0xC7 | (32 * (a10 & 1)) | (8 * (a8 & 1)) | (16 * (a9 & 1));
  return v13;
}
