// Function: sub_2353240
// Address: 0x2353240
//
__int64 __fastcall sub_2353240(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 *v13; // r13
  __int64 v14; // rdx
  int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  void *v25; // rdi
  unsigned int v26; // r14d
  void *v27; // rdi
  int v28; // eax
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 result; // rax
  unsigned int v32; // r13d
  __int64 *v33; // rax
  const void *v34; // rsi
  size_t v35; // rdx
  __int64 *v36; // rax
  const void *v37; // rsi
  int v38; // eax

  v7 = *a2;
  *(_QWORD *)(a1 + 48) = 1;
  *(_QWORD *)a1 = v7;
  *(_DWORD *)(a1 + 8) = *((_DWORD *)a2 + 2);
  *(_WORD *)(a1 + 12) = *((_WORD *)a2 + 6);
  *(_QWORD *)(a1 + 16) = a2[2];
  *(_QWORD *)(a1 + 24) = a2[3];
  *(_QWORD *)(a1 + 32) = a2[4];
  v9 = a2[5];
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_DWORD *)(a1 + 72) = 0;
  v10 = a2[7];
  *(_QWORD *)(a1 + 40) = v9;
  LODWORD(v9) = *((_DWORD *)a2 + 18);
  *(_QWORD *)(a1 + 56) = v10;
  v11 = a2[8];
  *(_DWORD *)(a1 + 72) = v9;
  *(_QWORD *)(a1 + 64) = v11;
  ++a2[6];
  a2[7] = 0;
  a2[8] = 0;
  *((_DWORD *)a2 + 18) = 0;
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)(a1 + 88) = 0;
  v12 = *((unsigned int *)a2 + 22);
  if ( (_DWORD)v12 )
    sub_2303CE0(a1 + 80, (char **)a2 + 10, v12, a4, a5, a6);
  v13 = (__int64 *)(a1 + 504);
  *(_QWORD *)(a1 + 96) = a2[12];
  *(_QWORD *)(a1 + 104) = a2[13];
  *(_QWORD *)(a1 + 112) = a2[14];
  *(_QWORD *)(a1 + 120) = a2[15];
  *(_BYTE *)(a1 + 128) = *((_BYTE *)a2 + 128);
  sub_278A460(a1 + 136, a2 + 17);
  *(_QWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 368) = 0;
  *(_DWORD *)(a1 + 376) = 0;
  v14 = a2[45];
  v15 = *((_DWORD *)a2 + 94);
  ++a2[44];
  *(_QWORD *)(a1 + 360) = v14;
  v16 = a2[46];
  *(_DWORD *)(a1 + 376) = v15;
  *(_QWORD *)(a1 + 352) = 1;
  *(_QWORD *)(a1 + 368) = v16;
  a2[45] = 0;
  a2[46] = 0;
  *((_DWORD *)a2 + 94) = 0;
  sub_234E5E0(a1 + 384, (__int64)(a2 + 48), v16, v17, v18, v19);
  v20 = a2[60];
  *(_QWORD *)(a1 + 488) = 0;
  *(_QWORD *)(a1 + 496) = 1;
  *(_QWORD *)(a1 + 480) = v20;
  do
  {
    if ( v13 )
      *v13 = -4096;
    v13 += 2;
  }
  while ( v13 != (__int64 *)(a1 + 568) );
  sub_23530E0(a1 + 488, (__int64)(a2 + 61));
  v25 = (void *)(a1 + 584);
  *(_QWORD *)(a1 + 568) = a1 + 584;
  *(_QWORD *)(a1 + 576) = 0x400000000LL;
  v26 = *((_DWORD *)a2 + 144);
  if ( v26 && v13 != a2 + 71 )
  {
    v36 = (__int64 *)a2[71];
    v37 = a2 + 73;
    if ( v36 == a2 + 73 )
    {
      v21 = 16LL * v26;
      if ( v26 <= 4
        || (sub_C8D5F0((__int64)v13, (const void *)(a1 + 584), v26, 0x10u, v26, v24),
            v25 = *(void **)(a1 + 568),
            v37 = (const void *)a2[71],
            (v21 = 16LL * *((unsigned int *)a2 + 144)) != 0) )
      {
        memcpy(v25, v37, v21);
      }
      *(_DWORD *)(a1 + 576) = v26;
      *((_DWORD *)a2 + 144) = 0;
    }
    else
    {
      *(_QWORD *)(a1 + 568) = v36;
      v38 = *((_DWORD *)a2 + 145);
      *(_DWORD *)(a1 + 576) = v26;
      *(_DWORD *)(a1 + 580) = v38;
      a2[71] = (__int64)v37;
      a2[72] = 0;
    }
  }
  *(_QWORD *)(a1 + 648) = a1 + 664;
  *(_QWORD *)(a1 + 656) = 0x800000000LL;
  if ( *((_DWORD *)a2 + 164) )
    sub_2303B80(a1 + 648, (char **)a2 + 81, v21, v22, v23, v24);
  *(_QWORD *)(a1 + 736) = 0;
  v27 = (void *)(a1 + 784);
  *(_QWORD *)(a1 + 744) = 0;
  *(_DWORD *)(a1 + 752) = 0;
  v28 = *((_DWORD *)a2 + 188);
  v29 = a2[92];
  ++a2[91];
  *(_DWORD *)(a1 + 752) = v28;
  LOBYTE(v28) = *((_BYTE *)a2 + 760);
  *(_QWORD *)(a1 + 736) = v29;
  v30 = a2[93];
  *(_BYTE *)(a1 + 760) = v28;
  result = 0x400000000LL;
  a2[92] = 0;
  a2[93] = 0;
  *((_DWORD *)a2 + 188) = 0;
  *(_QWORD *)(a1 + 728) = 1;
  *(_QWORD *)(a1 + 744) = v30;
  *(_QWORD *)(a1 + 768) = a1 + 784;
  *(_QWORD *)(a1 + 776) = 0x400000000LL;
  v32 = *((_DWORD *)a2 + 194);
  if ( v32 )
  {
    result = (__int64)(a2 + 96);
    if ( (__int64 *)(a1 + 768) != a2 + 96 )
    {
      v33 = (__int64 *)a2[96];
      v34 = a2 + 98;
      if ( v33 == a2 + 98 )
      {
        v35 = 16LL * v32;
        if ( v32 <= 4
          || (result = sub_C8D5F0(a1 + 768, (const void *)(a1 + 784), v32, 0x10u, a1 + 768, v32),
              v27 = *(void **)(a1 + 768),
              v34 = (const void *)a2[96],
              (v35 = 16LL * *((unsigned int *)a2 + 194)) != 0) )
        {
          result = (__int64)memcpy(v27, v34, v35);
        }
        *(_DWORD *)(a1 + 776) = v32;
        *((_DWORD *)a2 + 194) = 0;
      }
      else
      {
        *(_QWORD *)(a1 + 768) = v33;
        result = *((unsigned int *)a2 + 195);
        *(_DWORD *)(a1 + 776) = v32;
        *(_DWORD *)(a1 + 780) = result;
        a2[96] = (__int64)v34;
        a2[97] = 0;
      }
    }
  }
  return result;
}
