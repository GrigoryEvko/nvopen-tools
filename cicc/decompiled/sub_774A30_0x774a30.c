// Function: sub_774A30
// Address: 0x774a30
//
__int64 __fastcall sub_774A30(__int64 a1, char a2)
{
  __int64 *v2; // rax
  _QWORD *v4; // rax
  _QWORD *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rcx
  int v10; // esi
  int v11; // eax
  __int64 v12; // rax
  int v13; // r8d
  char v14; // r12
  bool v15; // al
  int v16; // edx
  int v17; // eax
  __int64 result; // rax
  unsigned int v19; // eax
  _DWORD *v20; // rdx

  v2 = (__int64 *)qword_4F08338;
  if ( qword_4F08338 )
  {
    *(_QWORD *)a1 = qword_4F08338;
    qword_4F08338 = *v2;
    v4 = *(_QWORD **)a1;
  }
  else
  {
    v4 = (_QWORD *)sub_823970(128);
    *(_QWORD *)a1 = v4;
  }
  *v4 = 0;
  v4[15] = 0;
  memset(
    (void *)((unsigned __int64)(v4 + 1) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)v4 - (((_DWORD)v4 + 8) & 0xFFFFFFF8) + 128) >> 3));
  *(_QWORD *)(a1 + 8) = 7;
  *(_QWORD *)(a1 + 24) = 0;
  v5 = (_QWORD *)qword_4F082A0;
  if ( qword_4F082A0 )
  {
    qword_4F082A0 = *(_QWORD *)(qword_4F082A0 + 8);
    v6 = 0;
  }
  else
  {
    v5 = (_QWORD *)sub_823970(0x10000);
    v6 = *(_QWORD *)(a1 + 24);
  }
  *v5 = v6;
  *(_QWORD *)(a1 + 24) = v5;
  v5[1] = 0;
  v7 = *(_QWORD *)(a1 + 24);
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 16) = v7 + 24;
  v8 = qword_4F082D8;
  *(_DWORD *)(a1 + 40) = 1;
  if ( v8 )
  {
    *(_QWORD *)(a1 + 56) = v8;
    qword_4F082D8 = *(_QWORD *)v8;
  }
  else
  {
    v8 = sub_823970(32);
    *(_QWORD *)(a1 + 56) = v8;
  }
  *(_OWORD *)v8 = 0;
  *(_OWORD *)(v8 + 16) = 0;
  v9 = *(_QWORD *)(a1 + 56);
  *(_QWORD *)(a1 + 64) = 7;
  v10 = *(_DWORD *)(v9 + 4);
  *(_DWORD *)(v9 + 4) = 1;
  if ( v10 )
  {
    LOBYTE(v19) = 1;
    do
    {
      v19 = ((_BYTE)v19 + 1) & 7;
      v20 = (_DWORD *)(v9 + 4LL * v19);
    }
    while ( *v20 );
    *v20 = v10;
  }
  v11 = *(_DWORD *)(a1 + 68) + 1;
  *(_DWORD *)(a1 + 68) = v11;
  if ( (unsigned int)(2 * v11) > 7 )
    sub_7702C0(a1 + 56);
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  v12 = *(_QWORD *)&dword_4F077C8;
  v13 = dword_4D0488C;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 112) = v12;
  LOBYTE(v12) = *(_BYTE *)(a1 + 132);
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_DWORD *)(a1 + 128) = 1;
  v14 = v12 & 0xF0 | a2 & 1;
  v15 = 0;
  *(_BYTE *)(a1 + 132) = v14;
  if ( !v13 )
  {
    v15 = 1;
    if ( word_4D04898 )
    {
      if ( (_DWORD)qword_4F077B4 && qword_4F077A0 > 0x765Bu )
        v15 = !sub_729F80(dword_4F063F8);
    }
  }
  v16 = 0;
  *(_WORD *)(a1 + 132) = *(_WORD *)(a1 + 132) & 0xFE0F | (16 * v15);
  if ( dword_4F077BC )
    LOBYTE(v16) = (_DWORD)qword_4F077B4 == 0;
  v17 = *(unsigned __int8 *)(a1 + 133);
  *(_DWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  ++qword_4F082A8;
  result = (2 * v16) | v17 & 0xFFFFFFC1;
  *(_BYTE *)(a1 + 133) = result;
  return result;
}
