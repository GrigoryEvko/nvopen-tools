// Function: sub_82B9D0
// Address: 0x82b9d0
//
__int64 __fastcall sub_82B9D0(__int64 a1, __int64 a2, __int64 a3, char a4, __int64 a5, __int64 a6, __int64 *a7)
{
  char v8; // si
  __int64 v11; // r12
  char v12; // al
  __int64 *v13; // r9
  _QWORD *v14; // rdx
  __int64 v15; // rax
  __int64 result; // rax
  __int64 v17; // rax
  __int64 v18; // [rsp+0h] [rbp-40h]

  v8 = a4;
  v11 = a1;
  v12 = *(_BYTE *)(a1 + 80);
  v13 = a7;
  if ( v12 == 16 )
  {
    v11 = **(_QWORD **)(a1 + 88);
    v12 = *(_BYTE *)(v11 + 80);
  }
  if ( v12 == 24 )
    v11 = *(_QWORD *)(v11 + 88);
  v14 = qword_4D03C68;
  if ( qword_4D03C68 )
  {
    qword_4D03C68 = (_QWORD *)*qword_4D03C68;
  }
  else
  {
    v18 = a5;
    v17 = sub_823970(152);
    v13 = a7;
    a5 = v18;
    v8 = a4;
    v14 = (_QWORD *)v17;
  }
  *v14 = 0;
  v14[18] = 0;
  memset(
    (void *)((unsigned __int64)(v14 + 1) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)v14 - (((_DWORD)v14 + 8) & 0xFFFFFFF8) + 152) >> 3));
  v14[1] = a1;
  v15 = *(_QWORD *)(v11 + 88);
  if ( *(_BYTE *)(v11 + 80) == 20 )
    v15 = *(_QWORD *)(v15 + 176);
  if ( (*(_BYTE *)(v15 + 194) & 0x40) != 0 )
    *((_BYTE *)v14 + 145) |= 0x20u;
  v14[3] = a2;
  *((_BYTE *)v14 + 32) = 1;
  *((_BYTE *)v14 + 33) = v8;
  v14[5] = a5;
  v14[14] = a3;
  v14[15] = a6;
  result = *v13;
  *v14 = *v13;
  *v13 = (__int64)v14;
  return result;
}
