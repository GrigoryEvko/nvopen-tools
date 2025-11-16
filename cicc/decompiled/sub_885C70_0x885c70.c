// Function: sub_885C70
// Address: 0x885c70
//
void __fastcall sub_885C70(__int64 a1, __int64 a2, _QWORD *a3, char a4, __int64 a5, __int64 a6, _QWORD **a7, int a8)
{
  __m128i v11; // xmm3
  __int64 v12; // rax
  int v13; // r9d
  int v14; // ecx
  __int64 v15; // rbx
  int v16; // esi
  __int64 v17; // rax
  __int64 v18; // rax
  _QWORD *v19; // rax
  int v20; // ecx
  int v21; // [rsp+Ch] [rbp-44h]
  int v22; // [rsp+10h] [rbp-40h]
  int v24; // [rsp+14h] [rbp-3Ch]
  int v25; // [rsp+14h] [rbp-3Ch]

  if ( (*(_BYTE *)(a1 + 17) & 0x20) != 0 )
  {
    if ( !a2 )
      return;
    v15 = qword_4F60018;
    v13 = 0;
    v14 = 0;
    v16 = 1;
    if ( !qword_4F60018 )
      goto LABEL_12;
    goto LABEL_15;
  }
  if ( a8 && (unsigned int)sub_866580() )
  {
    v13 = 1;
    v14 = 0;
    goto LABEL_9;
  }
  if ( !sub_87E2B0((_QWORD *)a1, *(_QWORD **)(a5 + 8)) )
  {
LABEL_8:
    v13 = 0;
    v14 = 0;
    goto LABEL_9;
  }
  if ( dword_4F077BC && qword_4F077A8 <= 0x9D6Bu )
  {
    v14 = 1;
    v13 = qword_4F077B4;
    if ( !(_DWORD)qword_4F077B4 )
    {
LABEL_9:
      if ( a2 )
        goto LABEL_11;
      goto LABEL_10;
    }
    goto LABEL_7;
  }
  if ( !dword_4F077C0 || a2 || (v13 = qword_4F077B4) != 0 )
  {
LABEL_7:
    sub_6851C0(0x64u, dword_4F07508);
    *(__m128i *)a1 = _mm_loadu_si128(xmmword_4F06660);
    *(__m128i *)(a1 + 16) = _mm_loadu_si128(&xmmword_4F06660[1]);
    *(__m128i *)(a1 + 32) = _mm_loadu_si128(&xmmword_4F06660[2]);
    v11 = _mm_loadu_si128(&xmmword_4F06660[3]);
    v12 = *(_QWORD *)dword_4F07508;
    *(_BYTE *)(a1 + 17) |= 0x20u;
    *(_QWORD *)(a1 + 8) = v12;
    *(__m128i *)(a1 + 48) = v11;
    goto LABEL_8;
  }
  v14 = 1;
LABEL_10:
  if ( (*(_BYTE *)(a1 + 17) & 0x20) != 0 )
    return;
LABEL_11:
  v15 = qword_4F60018;
  v16 = 0;
  if ( !qword_4F60018 )
  {
LABEL_12:
    v21 = v13;
    v22 = v14;
    v17 = sub_823970(128);
    v13 = v21;
    v14 = v22;
    v15 = v17;
    goto LABEL_16;
  }
LABEL_15:
  qword_4F60018 = *(_QWORD *)v15;
LABEL_16:
  *(_QWORD *)v15 = 0;
  *(_BYTE *)(v15 + 42) &= 0xE0u;
  v18 = *(_QWORD *)&dword_4F077C8;
  *(_QWORD *)(v15 + 8) = 0;
  *(_QWORD *)(v15 + 16) = 0;
  *(_QWORD *)(v15 + 24) = 0;
  *(_QWORD *)(v15 + 32) = v18;
  *(_WORD *)(v15 + 40) = 0;
  *(_QWORD *)(v15 + 48) = 0;
  *(_QWORD *)(v15 + 56) = 0;
  *(_QWORD *)(v15 + 64) = v18;
  *(_QWORD *)(v15 + 72) = v18;
  *(_QWORD *)(v15 + 80) = v18;
  *(_QWORD *)(v15 + 88) = v18;
  *(_QWORD *)(v15 + 96) = v18;
  *(_QWORD *)(v15 + 104) = v18;
  *(_QWORD *)(v15 + 112) = v18;
  *(_DWORD *)(v15 + 120) = 0;
  if ( a2 )
  {
    *(_QWORD *)(v15 + 16) = a2;
    *(_QWORD *)(v15 + 32) = *a3;
    *(_BYTE *)(v15 + 40) = a4;
    if ( !v16 )
    {
      v24 = v14;
      if ( v13 )
      {
        v19 = sub_87EBB0(0x12u, *(_QWORD *)a1, (_QWORD *)(a1 + 8));
        v20 = v24;
        *((_BYTE *)v19 + 81) = *(_BYTE *)(a1 + 17) & 0x20 | *((_BYTE *)v19 + 81) & 0xDF;
        *(_QWORD *)(a1 + 24) = v19;
      }
      else
      {
        v19 = sub_885AD0(0x12u, a1, dword_4F04C64, v14);
        v20 = v24;
      }
      *((_BYTE *)v19 + 84) = (32 * (a8 & 1)) | *((_BYTE *)v19 + 84) & 0xDF;
      if ( dword_4D047B8 )
        *((_BYTE *)v19 + 83) |= 0x40u;
      goto LABEL_22;
    }
  }
  else if ( !v16 )
  {
    v25 = v14;
    v19 = sub_87EBB0(0x12u, *(_QWORD *)a1, (_QWORD *)(a1 + 8));
    *(_BYTE *)(a1 + 16) &= ~1u;
    v20 = v25;
    *(_QWORD *)(a1 + 24) = v19;
LABEL_22:
    *(_QWORD *)(v15 + 8) = v19;
    v19[11] = v15;
    if ( v20 )
      *((_WORD *)v19 + 41) |= 0x4004u;
    *((_DWORD *)v19 + 11) = ++dword_4F066AC;
  }
  *(_QWORD *)(v15 + 48) = a6;
  if ( *(_QWORD *)(a5 + 8) )
  {
    **a7 = v15;
  }
  else
  {
    *(_QWORD *)(a5 + 8) = v15;
    *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 752) = v15;
  }
  *a7 = (_QWORD *)v15;
}
