// Function: sub_84CF20
// Address: 0x84cf20
//
__int64 __fastcall sub_84CF20(__int64 a1, int a2, int a3, int a4, __int64 a5, FILE *a6, const __m128i **a7, _DWORD *a8)
{
  char v10; // al
  __int64 v11; // r13
  unsigned int v12; // r14d
  __int64 v13; // r15
  __int64 v14; // rdi
  _QWORD *v15; // r14
  __int64 v16; // rdx
  char v17; // al
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 v20; // r13
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // r14
  __int64 v27; // rbx
  int v28; // eax
  const __m128i *v30; // rax
  const __m128i *v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // [rsp+0h] [rbp-60h]
  __int64 v34; // [rsp+0h] [rbp-60h]
  __int64 v35; // [rsp+0h] [rbp-60h]
  const __m128i *v36; // [rsp+8h] [rbp-58h]
  int v39; // [rsp+1Ch] [rbp-44h]
  unsigned int v40; // [rsp+20h] [rbp-40h] BYREF
  int v41; // [rsp+24h] [rbp-3Ch] BYREF
  __int64 v42[7]; // [rsp+28h] [rbp-38h] BYREF

  v10 = *(_BYTE *)(a1 + 140) & 0xFB;
  v42[0] = 0;
  v40 = 0;
  v41 = 0;
  v39 = 0;
  if ( v10 == 8 )
    v39 = sub_8D4C10(a1, dword_4F077C4 != 2);
  v11 = sub_8D2290(a1);
  v12 = sub_8D3F60(v11);
  if ( !v12 )
  {
    *a7 = (const __m128i *)sub_72C930();
    *a8 = 0;
    return v12;
  }
  if ( !a5 )
  {
    v15 = 0;
    goto LABEL_8;
  }
  v13 = 0;
  if ( !a3 && *(_BYTE *)(a5 + 8) == 1 )
  {
    v15 = *(_QWORD **)(a5 + 24);
    if ( !v15 )
      goto LABEL_8;
    v13 = a5;
    a5 = *(_QWORD *)(a5 + 24);
  }
  v14 = a5;
  v15 = (_QWORD *)a5;
  a5 = v13;
  if ( (unsigned int)sub_82ED80(v14) )
  {
LABEL_36:
    v12 = 0;
    *a8 = 1;
    return v12;
  }
LABEL_8:
  v16 = *(_QWORD *)(*(_QWORD *)(v11 + 168) + 32LL);
  v17 = *(_BYTE *)(v16 + 81);
  if ( (v17 & 0x40) != 0 )
    goto LABEL_36;
  if ( (v17 & 0x10) != 0 )
  {
    v35 = *(_QWORD *)(*(_QWORD *)(v11 + 168) + 32LL);
    v28 = sub_8DBE70(*(_QWORD *)(v16 + 64));
    v16 = v35;
    if ( v28 )
      goto LABEL_36;
  }
  v18 = *(_QWORD *)(v16 + 88);
  if ( (*(_BYTE *)(v18 + 267) & 1) == 0 )
    goto LABEL_16;
  v19 = *(_QWORD *)(v18 + 88);
  if ( v19 )
  {
    if ( (*(_BYTE *)(v18 + 160) & 1) != 0 )
      v19 = v16;
  }
  else
  {
    v19 = v16;
  }
  if ( (*(_BYTE *)(v19 + 81) & 2) != 0 && (*(_BYTE *)(v18 + 267) & 2) != 0 )
  {
LABEL_16:
    v33 = v16;
    sub_8C2270(v16);
    v16 = v33;
  }
  v20 = 0;
  if ( unk_4D04864 )
  {
    v21 = *(_QWORD *)(v16 + 88);
    v22 = *(_QWORD *)(v21 + 88);
    if ( v22 )
    {
      if ( (*(_BYTE *)(v21 + 160) & 1) != 0 )
        v22 = v16;
    }
    else
    {
      v22 = v16;
    }
    v20 = 0;
    if ( (*(_BYTE *)(v22 + 81) & 2) != 0 && *(char *)(v18 + 266) >= 0 )
    {
      if ( a5 || dword_4D0478C && a3 )
      {
        v34 = v16;
        v23 = sub_829A30(v16, v15);
        v16 = v34;
        v20 = v23;
      }
      else
      {
        v20 = 0;
      }
    }
  }
  v24 = *(_QWORD *)(v18 + 216);
  if ( v24 )
  {
    v25 = sub_84AC10(v24, 0, 0, 1, 0, v15, (_QWORD *)a5, a2 == 0, 0, 0, 0, 7, a6, 0, 0, &v41, &v40, 0, 0, v42);
    v26 = v25;
    if ( v25 )
    {
      v27 = *(_QWORD *)(v25 + 88);
      v36 = *(const __m128i **)(*(_QWORD *)(v27 + 152) + 160LL);
      if ( (unsigned int)sub_8DBE70(v36) )
      {
        v12 = 0;
        *a7 = (const __m128i *)sub_72C930();
      }
      else
      {
        *a7 = v36;
        if ( a4 )
        {
          v30 = (const __m128i *)sub_7259C0(12);
          v31 = *a7;
          v30[11].m128i_i8[8] = 4;
          v30[10].m128i_i64[0] = (__int64)v31;
          *a7 = v30;
        }
        if ( v39 )
          *a7 = sub_73C570(*a7, v39);
        *a8 = 0;
        if ( *(char *)(v27 + 193) >= 0 || a2 )
        {
          v12 = 1;
        }
        else
        {
          v32 = v26;
          v12 = 1;
          sub_6854C0(0xB4Bu, a6, v32);
        }
      }
      goto LABEL_44;
    }
  }
  else
  {
    sub_6854C0(0xB95u, a6, v16);
  }
  v12 = v40;
  if ( v40 )
  {
    v12 = 0;
    *a8 = 1;
  }
  else
  {
    *a7 = (const __m128i *)sub_72C930();
    *a8 = 0;
  }
LABEL_44:
  if ( v20 )
    sub_8792A0(v20, v18 + 216);
  return v12;
}
