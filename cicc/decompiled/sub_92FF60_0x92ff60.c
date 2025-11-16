// Function: sub_92FF60
// Address: 0x92ff60
//
__int64 __fastcall sub_92FF60(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rbx
  char v6; // r14
  __int64 v7; // r15
  const __m128i *v8; // rbx
  __int64 v9; // r13
  __int64 v10; // rax
  unsigned __int64 v11; // rsi
  __int64 v12; // r14
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rax
  unsigned __int16 v16; // cx
  unsigned int v17; // r11d
  __int64 v18; // rsi
  __int64 v19; // rax
  unsigned int v20; // r11d
  _BOOL4 v21; // ebx
  int v22; // eax
  __int64 v23; // rcx
  _BOOL4 v24; // r10d
  int v25; // r13d
  unsigned int v26; // r11d
  int v28; // r10d
  __int64 v29; // rdi
  char v30; // dl
  unsigned __int64 v31; // rax
  __int64 v32; // r13
  __int64 v33; // rax
  int v34; // ebx
  int v35; // r15d
  __int64 v36; // rax
  int v37; // eax
  __int64 v38; // rdi
  _BOOL4 v39; // r14d
  unsigned int v40; // r13d
  __int64 v41; // rax
  __int64 v42; // rdi
  _BOOL4 v43; // ebx
  unsigned int v44; // eax
  char v45; // al
  __int64 v46; // rdi
  char v47; // al
  __m128i *v48; // [rsp-10h] [rbp-90h]
  __int64 v49; // [rsp-10h] [rbp-90h]
  unsigned __int8 v50; // [rsp+Ch] [rbp-74h]
  unsigned int v51; // [rsp+Ch] [rbp-74h]
  unsigned int v52; // [rsp+10h] [rbp-70h]
  __int64 v53; // [rsp+10h] [rbp-70h]
  int v54; // [rsp+10h] [rbp-70h]
  __int64 v55; // [rsp+18h] [rbp-68h]
  int v56; // [rsp+18h] [rbp-68h]
  __int64 v57; // [rsp+18h] [rbp-68h]
  int v58; // [rsp+18h] [rbp-68h]
  const char *v59; // [rsp+20h] [rbp-60h] BYREF
  char v60; // [rsp+40h] [rbp-40h]
  char v61; // [rsp+41h] [rbp-3Fh]

  v5 = a2;
  v6 = *(_BYTE *)(a2 + 48);
  v7 = *(_QWORD *)(a2 + 8);
  switch ( v6 )
  {
    case 2:
      v8 = *(const __m128i **)(a2 + 56);
      v9 = sub_91FFE0((__int64)a1, v8, *(_QWORD *)(v7 + 120), a4);
      v10 = sub_9439D0(a1, v7);
      v11 = *(_QWORD *)(v7 + 120);
      v12 = v10;
      v55 = *(_QWORD *)(v10 + 8);
      if ( *(_QWORD *)(v9 + 8) == sub_91A390(a1[4] + 8LL, v11, 0, v13) )
      {
        v42 = *(_QWORD *)(v7 + 120);
        v43 = 0;
        if ( (*(_BYTE *)(v42 + 140) & 0xFB) == 8 )
        {
          v11 = dword_4F077C4 != 2;
          v43 = (sub_8D4C10(v42, v11) & 2) != 0;
        }
        v44 = sub_91CB50(v7, v11, v14);
        return sub_9472D0(a1, v9, v12, v44, v43);
      }
      else
      {
        LODWORD(v15) = sub_91CB50(v7, v11, v14);
        v16 = 0;
        v17 = v15;
        if ( (_DWORD)v15 )
        {
          _BitScanReverse64((unsigned __int64 *)&v15, (unsigned int)v15);
          LOBYTE(v16) = 63 - (v15 ^ 0x3F);
          HIBYTE(v16) = 1;
        }
        v61 = 1;
        v60 = 3;
        v18 = *(_QWORD *)(v9 + 8);
        v52 = v17;
        v59 = "consttmp";
        v19 = sub_921B80((__int64)a1, v18, (__int64)&v59, v16, 0);
        v20 = v52;
        v53 = v19;
        v51 = v20;
        sub_9472D0(a1, v9, v19, v20, 0);
        v48 = (__m128i *)&v8[4];
        v21 = 0;
        v22 = sub_92C9E0((__int64)a1, v53, 0, v55, 0, 0, v48);
        v23 = *(_QWORD *)(v7 + 120);
        v24 = 0;
        v25 = v22;
        v26 = v51;
        if ( (*(_BYTE *)(v23 + 140) & 0xFB) == 8 )
        {
          v57 = *(_QWORD *)(v7 + 120);
          v45 = sub_8D4C10(v57, dword_4F077C4 != 2);
          v46 = *(_QWORD *)(v7 + 120);
          v23 = v57;
          v26 = v51;
          v24 = 0;
          v21 = (v45 & 2) != 0;
          if ( (*(_BYTE *)(v46 + 140) & 0xFB) == 8 )
          {
            v47 = sub_8D4C10(v46, dword_4F077C4 != 2);
            v26 = v51;
            v23 = v57;
            v24 = (v47 & 2) != 0;
          }
        }
        sub_947440((_DWORD)a1, v12, v26, v24, v25, v26, v21, v23);
        return v49;
      }
    case 3:
      v38 = *(_QWORD *)(v7 + 120);
      v39 = 0;
      if ( (*(_BYTE *)(v38 + 140) & 0xFB) == 8 )
      {
        a2 = dword_4F077C4 != 2;
        v39 = (sub_8D4C10(v38, a2) & 2) != 0;
      }
      v40 = sub_91CB50(v7, a2, a3);
      v41 = sub_9439D0(a1, v7);
      return sub_947E80(a1, *(_QWORD *)(v5 + 56), v41, v40, v39);
    case 1:
      v28 = (_DWORD)a1 + 48;
      v29 = *(_QWORD *)(v7 + 120);
      if ( *(char *)(v29 + 142) < 0 )
      {
        v31 = *(unsigned int *)(v29 + 136);
      }
      else
      {
        v30 = *(_BYTE *)(v29 + 140);
        if ( v30 != 12 )
        {
          v31 = *(unsigned int *)(v29 + 136);
          if ( !*(_DWORD *)(v29 + 136) )
          {
            v6 = 0;
LABEL_16:
            v32 = *(_QWORD *)(v29 + 128);
            v54 = v28;
            v33 = sub_BCB2B0(a1[5]);
            v56 = sub_AD6530(v33);
            v34 = v50;
            v35 = sub_9439D0(a1, v7);
            BYTE1(v34) = v6;
            v36 = sub_BCB2E0(a1[15]);
            v37 = sub_ACD640(v36, v32, 0);
            return sub_B34240(v54, v35, v56, v37, v34, 0, 0, 0, 0);
          }
          goto LABEL_13;
        }
        v58 = v28;
        LODWORD(v31) = sub_8D4AB0(v29);
        v29 = *(_QWORD *)(v7 + 120);
        v28 = v58;
        v31 = (unsigned int)v31;
      }
      v30 = *(_BYTE *)(v29 + 140);
      if ( !v31 )
      {
        v6 = 0;
LABEL_14:
        if ( v30 == 12 )
        {
          do
            v29 = *(_QWORD *)(v29 + 160);
          while ( *(_BYTE *)(v29 + 140) == 12 );
        }
        goto LABEL_16;
      }
LABEL_13:
      _BitScanReverse64(&v31, v31);
      v50 = 63 - (v31 ^ 0x3F);
      goto LABEL_14;
    default:
      sub_91B8A0("unsupported dynamic initialization variant!", (_DWORD *)(v7 + 64), 1);
  }
}
