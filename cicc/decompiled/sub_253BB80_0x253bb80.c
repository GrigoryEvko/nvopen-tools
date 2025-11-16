// Function: sub_253BB80
// Address: 0x253bb80
//
__int64 __fastcall sub_253BB80(__int64 a1, __int64 a2)
{
  __int64 v2; // r9
  __int64 v3; // rcx
  __int64 v4; // rdi
  int v5; // eax
  unsigned __int64 v6; // r8
  unsigned __int64 v7; // r8
  __int64 *v8; // r11
  int v9; // r10d
  _QWORD *v10; // rdi
  _QWORD *v11; // rsi
  _QWORD *v12; // rdi
  _QWORD *v13; // rsi
  __int64 v14; // r8
  int v15; // r10d
  unsigned __int64 v16; // r8
  int v17; // eax
  __int64 v18; // rbp
  _QWORD v20[4]; // [rsp-20h] [rbp-20h] BYREF

  if ( !*(_BYTE *)(a1 + 96) || !*(_BYTE *)(a1 + 97) )
    return 0;
  v2 = a1;
  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)(a1 + 368);
  v5 = *(_DWORD *)(v2 + 384);
  if ( !v5 )
    JUMPOUT(0x253BAF7);
  if ( v3 != *(_QWORD *)(v4 + 8LL * ((v5 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4)))) )
    JUMPOUT(0x253BAE8);
  if ( *(_QWORD *)(v3 + 56) == a2 + 24 || (v6 = *(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    JUMPOUT(0x253BB30);
  v20[3] = v18;
  v7 = v6 - 24;
  v8 = v20;
  v9 = *(_DWORD *)(v2 + 232);
  v20[0] = v7;
  if ( v9 )
    goto LABEL_15;
LABEL_9:
  v10 = *(_QWORD **)(v2 + 248);
  v11 = &v10[*(unsigned int *)(v2 + 256)];
  if ( v11 == sub_2537F00(v10, (__int64)v11, v8) )
  {
    while ( 1 )
    {
      if ( *(_DWORD *)(v2 + 120) )
        JUMPOUT(0x253BA80);
      v12 = *(_QWORD **)(v2 + 136);
      v13 = &v12[*(unsigned int *)(v2 + 144)];
      if ( v13 != sub_2537F00(v12, (__int64)v13, v8) )
        break;
      if ( *(_QWORD *)(*(_QWORD *)(v14 + 40) + 56LL) == v14 + 24
        || (v16 = *(_QWORD *)(v14 + 24) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      {
        JUMPOUT(0x253BAE0);
      }
      v7 = v16 - 24;
      v20[0] = v7;
      if ( !v15 )
        goto LABEL_9;
LABEL_15:
      v17 = *(_DWORD *)(v2 + 240);
      if ( v17 )
      {
        if ( v7 != *(_QWORD *)(*(_QWORD *)(v2 + 224)
                             + 8LL * ((v17 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)))) )
          JUMPOUT(0x253BAFD);
        return 1;
      }
    }
  }
  return 1;
}
