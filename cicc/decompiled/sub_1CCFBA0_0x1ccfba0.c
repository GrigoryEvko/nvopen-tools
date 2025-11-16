// Function: sub_1CCFBA0
// Address: 0x1ccfba0
//
__int64 __fastcall sub_1CCFBA0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // rbx
  __int64 v12; // r14
  __int64 v13; // rax
  int v14; // eax
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // rbx
  _QWORD *v19; // rdi
  __int64 *v21; // rax
  __int64 j; // rdi
  unsigned __int8 v23; // r9
  __int64 *v24; // rax
  __int64 v25; // rax
  double v26; // xmm4_8
  double v27; // xmm5_8
  __int64 *v28; // rax
  __int64 i; // rdi
  __int64 *v30; // rax
  __int64 *v31; // rax
  __int64 v32; // rax
  double v33; // xmm4_8
  double v34; // xmm5_8
  __int64 *v35; // rax
  unsigned __int8 v36; // [rsp+1Fh] [rbp-31h]

  *(_DWORD *)(a1 + 168) = 0;
  v10 = *(_QWORD *)(a2 + 80);
  if ( v10 == a2 + 72 )
  {
    return 0;
  }
  else
  {
    v36 = 0;
    do
    {
      if ( !v10 )
        BUG();
      v12 = *(_QWORD *)(v10 + 24);
      if ( v10 + 16 != v12 )
      {
        while ( 1 )
        {
          if ( !v12 )
            BUG();
          if ( *(_BYTE *)(v12 - 8) != 78 )
            goto LABEL_12;
          v13 = *(_QWORD *)(v12 - 48);
          if ( *(_BYTE *)(v13 + 16) || (*(_BYTE *)(v13 + 33) & 0x20) == 0 )
            goto LABEL_12;
          v14 = *(_DWORD *)(v13 + 36);
          v15 = v12 - 24;
          if ( v14 == 4055 )
            break;
          if ( v14 == 4056 )
          {
            if ( (*(_BYTE *)(v12 - 1) & 0x40) != 0 )
              v28 = *(__int64 **)(v12 - 32);
            else
              v28 = (__int64 *)(v15 - 24LL * (*(_DWORD *)(v12 - 4) & 0xFFFFFFF));
            for ( i = *v28; *(_BYTE *)(i + 16) == 86; i = *(_QWORD *)(i - 24) )
              ;
            v23 = sub_1C2E970(i);
            if ( v23 )
            {
LABEL_45:
              v36 = v23;
              v31 = (__int64 *)sub_16498A0(v12 - 24);
              v32 = sub_159C4F0(v31);
              sub_1CCFA00(a1, v12 - 24, v32, a3, a4, a5, a6, v33, v34, a9, a10);
              goto LABEL_12;
            }
            if ( !(unsigned __int8)sub_1C2EAF0(i) && !(unsigned __int8)sub_1C2EA30(i) )
            {
LABEL_35:
              if ( !(unsigned __int8)sub_1C2E890(i) )
                goto LABEL_12;
            }
LABEL_26:
            v24 = (__int64 *)sub_16498A0(v12 - 24);
            v25 = sub_159C540(v24);
            goto LABEL_27;
          }
          if ( v14 == 4054 )
          {
            if ( (*(_BYTE *)(v12 - 1) & 0x40) != 0 )
              v21 = *(__int64 **)(v12 - 32);
            else
              v21 = (__int64 *)(v15 - 24LL * (*(_DWORD *)(v12 - 4) & 0xFFFFFFF));
            for ( j = *v21; *(_BYTE *)(j + 16) == 86; j = *(_QWORD *)(j - 24) )
              ;
            v23 = sub_1C2E890(j);
            if ( v23 )
              goto LABEL_45;
            if ( !(unsigned __int8)sub_1C2EAF0(j)
              && !(unsigned __int8)sub_1C2EA30(j)
              && !(unsigned __int8)sub_1C2E970(j) )
            {
              goto LABEL_12;
            }
            goto LABEL_26;
          }
LABEL_12:
          v12 = *(_QWORD *)(v12 + 8);
          if ( v10 + 16 == v12 )
            goto LABEL_13;
        }
        if ( (*(_BYTE *)(v12 - 1) & 0x40) != 0 )
          v30 = *(__int64 **)(v12 - 32);
        else
          v30 = (__int64 *)(v15 - 24LL * (*(_DWORD *)(v12 - 4) & 0xFFFFFFF));
        for ( i = *v30; *(_BYTE *)(i + 16) == 86; i = *(_QWORD *)(i - 24) )
          ;
        if ( !(unsigned __int8)sub_1C2EA30(i) && !(unsigned __int8)sub_1C2EAF0(i) )
        {
          if ( (unsigned __int8)sub_1C2E970(i) )
            goto LABEL_26;
          goto LABEL_35;
        }
        v35 = (__int64 *)sub_16498A0(v12 - 24);
        v25 = sub_159C4F0(v35);
LABEL_27:
        sub_1CCFA00(a1, v12 - 24, v25, a3, a4, a5, a6, v26, v27, a9, a10);
        v36 = 1;
        goto LABEL_12;
      }
LABEL_13:
      v10 = *(_QWORD *)(v10 + 8);
    }
    while ( a2 + 72 != v10 );
    v16 = *(unsigned int *)(a1 + 168);
    if ( (_DWORD)v16 )
    {
      v17 = 8 * v16;
      v18 = 0;
      do
      {
        v19 = *(_QWORD **)(*(_QWORD *)(a1 + 160) + v18);
        v18 += 8;
        sub_15F20C0(v19);
      }
      while ( v17 != v18 );
    }
  }
  return v36;
}
