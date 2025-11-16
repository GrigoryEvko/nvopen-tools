// Function: sub_21BD160
// Address: 0x21bd160
//
__int64 __fastcall sub_21BD160(
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
  __int64 v11; // rbx
  __int64 v12; // r15
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
  unsigned __int8 v24; // al
  __int64 *v25; // rax
  __int64 v26; // rax
  double v27; // xmm4_8
  double v28; // xmm5_8
  __int64 *v29; // rax
  __int64 i; // rdi
  __int64 *v31; // rax
  __int64 v32; // rax
  double v33; // xmm4_8
  double v34; // xmm5_8
  __int64 *v35; // rax
  __int64 k; // rdi
  __int64 *v37; // rax
  __int64 *v38; // rax
  unsigned __int8 v39; // [rsp+1Fh] [rbp-31h]

  v39 = sub_1636880(a1, a2);
  if ( v39 )
    return 0;
  *(_DWORD *)(a1 + 168) = 0;
  v11 = *(_QWORD *)(a2 + 80);
  if ( v11 == a2 + 72 )
  {
    return 0;
  }
  else
  {
    do
    {
      if ( !v11 )
        BUG();
      v12 = *(_QWORD *)(v11 + 24);
      if ( v12 != v11 + 16 )
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
              v29 = *(__int64 **)(v12 - 32);
            else
              v29 = (__int64 *)(v15 - 24LL * (*(_DWORD *)(v12 - 4) & 0xFFFFFFF));
            for ( i = *v29; *(_BYTE *)(i + 16) == 86; i = *(_QWORD *)(i - 24) )
              ;
            v23 = sub_1C2E970(i);
            if ( v23 )
            {
LABEL_43:
              v39 = v23;
              v37 = (__int64 *)sub_16498A0(v12 - 24);
              v26 = sub_159C4F0(v37);
              goto LABEL_44;
            }
            if ( !(unsigned __int8)sub_1C2EAF0(i)
              && !(unsigned __int8)sub_1C2EA30(i)
              && !(unsigned __int8)sub_1C2E890(i) )
            {
              goto LABEL_12;
            }
LABEL_32:
            v31 = (__int64 *)sub_16498A0(v12 - 24);
            v32 = sub_159C540(v31);
            goto LABEL_33;
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
              goto LABEL_43;
            v24 = sub_1C2EBB0(j);
            if ( v24 )
            {
              v39 = v24;
              v25 = (__int64 *)sub_16498A0(v12 - 24);
              v26 = sub_159C540(v25);
LABEL_44:
              sub_21BCFC0(a1, v12 - 24, v26, a3, a4, a5, a6, v27, v28, a9, a10);
            }
          }
LABEL_12:
          v12 = *(_QWORD *)(v12 + 8);
          if ( v11 + 16 == v12 )
            goto LABEL_13;
        }
        if ( (*(_BYTE *)(v12 - 1) & 0x40) != 0 )
          v35 = *(__int64 **)(v12 - 32);
        else
          v35 = (__int64 *)(v15 - 24LL * (*(_DWORD *)(v12 - 4) & 0xFFFFFFF));
        for ( k = *v35; *(_BYTE *)(k + 16) == 86; k = *(_QWORD *)(k - 24) )
          ;
        if ( !(unsigned __int8)sub_1C2EA30(k) && !(unsigned __int8)sub_1C2EAF0(k) )
        {
          if ( !(unsigned __int8)sub_1C2E970(k) && !(unsigned __int8)sub_1C2E890(k) )
            goto LABEL_12;
          goto LABEL_32;
        }
        v38 = (__int64 *)sub_16498A0(v12 - 24);
        v32 = sub_159C4F0(v38);
LABEL_33:
        sub_21BCFC0(a1, v12 - 24, v32, a3, a4, a5, a6, v33, v34, a9, a10);
        v39 = 1;
        goto LABEL_12;
      }
LABEL_13:
      v11 = *(_QWORD *)(v11 + 8);
    }
    while ( a2 + 72 != v11 );
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
  return v39;
}
