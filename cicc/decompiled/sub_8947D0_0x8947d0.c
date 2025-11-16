// Function: sub_8947D0
// Address: 0x8947d0
//
__int64 __fastcall sub_8947D0(_BYTE *a1)
{
  __int64 v2; // rdx
  __int64 v3; // r14
  __int64 v4; // rbx
  _QWORD *v5; // r13
  int v6; // eax
  __int64 v7; // r15
  __int64 v8; // r10
  int v9; // edx
  unsigned int v10; // edx
  const __m128i *v11; // rdi
  unsigned int *v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  unsigned __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 *v22; // r9
  unsigned int *v24; // [rsp+0h] [rbp-2D0h]
  __int64 v25; // [rsp+0h] [rbp-2D0h]
  __int64 v26; // [rsp+8h] [rbp-2C8h]
  __int64 v27; // [rsp+18h] [rbp-2B8h]
  unsigned int v28; // [rsp+28h] [rbp-2A8h]
  __int16 v29; // [rsp+2Eh] [rbp-2A2h]
  unsigned __int16 v30; // [rsp+30h] [rbp-2A0h]
  __int16 v31; // [rsp+32h] [rbp-29Eh]
  int v32; // [rsp+34h] [rbp-29Ch]
  int v33; // [rsp+38h] [rbp-298h]
  int v34; // [rsp+3Ch] [rbp-294h]
  __int64 v35; // [rsp+40h] [rbp-290h]
  unsigned int v36; // [rsp+40h] [rbp-290h]
  __int64 v37; // [rsp+48h] [rbp-288h]
  int v38; // [rsp+5Ch] [rbp-274h] BYREF
  _BYTE v39[96]; // [rsp+60h] [rbp-270h] BYREF
  _QWORD v40[66]; // [rsp+C0h] [rbp-210h] BYREF

  v2 = *(_QWORD *)(*(_QWORD *)a1 + 96LL);
  *(_BYTE *)(v2 + 48) |= 1u;
  v3 = *(_QWORD *)(v2 + 24);
  v4 = *((_QWORD *)a1 + 22);
  v35 = v2;
  v5 = *(_QWORD **)(*(_QWORD *)(v3 + 96) + 32LL);
  v6 = sub_8D0B70(v3);
  v7 = *(_QWORD *)a1;
  v33 = v6;
  v37 = *(_QWORD *)(*(_QWORD *)a1 + 64LL);
  if ( (*(_BYTE *)(v37 + 177) & 0x20) != 0 )
    a1[162] |= 0x40u;
  *(_QWORD *)(v4 + 24) = v5[13];
  *(_QWORD *)(v35 + 40) = *(_QWORD *)&dword_4F063F8;
  v36 = dword_4F063F8;
  v30 = word_4F063FC[0];
  v34 = dword_4F07508[0];
  v29 = dword_4F07508[1];
  v32 = dword_4F061D8;
  v31 = unk_4F061DC;
  v8 = sub_892400((__int64)v5);
  if ( *(_QWORD *)(v8 + 8) )
  {
    v9 = -((a1[162] & 0x40) == 0);
    memset(v40, 0, 0x1D8u);
    v40[19] = v40;
    v10 = (v9 & 0xFFFFFFFC) + 2052;
    v40[3] = *(_QWORD *)&dword_4F063F8;
    if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
      BYTE2(v40[22]) |= 1u;
    v27 = v8;
    sub_864700(*(_QWORD *)(v8 + 32), 0, 0, v7, v3, 0, 1, v10);
    v11 = (const __m128i *)v5[7];
    v28 = dword_4F04C3C;
    dword_4F04C3C = 1;
    sub_854C10(v11);
    sub_7BC160(v27);
    sub_8756F0(32770, v7, (_QWORD *)(v7 + 48), 0);
    memset(v39, 0, 0x58u);
    v38 = 0;
    sub_66DF40((__int64)a1, (__int64)v40, 0, 0, v37, &v38, (__int64)v39);
    v12 = v24;
    if ( (a1[162] & 0x40) != 0 )
    {
      sub_854B40();
    }
    else
    {
      v12 = 0;
      sub_854980(v7, 0);
    }
    v17 = v5[16];
    if ( v17 )
    {
      v18 = sub_5CF220((const __m128i *)v17, 0, v3, 0, 0, v37, 0, 0);
      v17 = (unsigned __int64)v40;
      v12 = (unsigned int *)1;
      v40[25] = v18;
      sub_644920(v40, 1);
      v13 = v25;
      v14 = v26;
    }
    if ( word_4F06418[0] != 9 )
    {
      v12 = &dword_4F063F8;
      v17 = 65;
      sub_6851C0(0x41u, &dword_4F063F8);
      while ( word_4F06418[0] != 9 )
        sub_7B8B50(0x41u, &dword_4F063F8, v13, v14, v15, v16);
    }
    sub_7B8B50(v17, v12, v13, v14, v15, v16);
    dword_4F04C3C = v28;
    sub_863FE0(v17, v28, v19, v20, v21, v22);
  }
  dword_4F07508[0] = v34;
  LOWORD(dword_4F07508[1]) = v29;
  dword_4F063F8 = v36;
  word_4F063FC[0] = v30;
  dword_4F061D8 = v32;
  unk_4F061DC = v31;
  if ( v33 )
    sub_8D0B10();
  return sub_8CBA30(a1);
}
