// Function: sub_76ECE0
// Address: 0x76ece0
//
__int64 __fastcall sub_76ECE0(__int64 a1, _DWORD *a2)
{
  char v3; // bl
  __int64 result; // rax
  const __m128i *v5; // rdi
  const __m128i *v6; // r15
  const __m128i *v7; // r14
  int v8; // eax
  const __m128i *v9; // r8
  const __m128i *v10; // rax
  const __m128i *v11; // rdi
  int v12; // eax
  const __m128i *v13; // rax
  _QWORD *v14; // rbx
  const __m128i *v15; // r14
  __int64 v16; // r9
  __int64 v17; // rbx
  int v18; // eax
  const __m128i *v19; // rdi
  __int64 v20; // rax
  _QWORD *v21; // r15
  _QWORD *v22; // rax
  const __m128i *v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  _UNKNOWN *__ptr32 *v27; // r8
  __int64 v28; // [rsp+8h] [rbp-48h]
  __int64 v29; // [rsp+8h] [rbp-48h]
  __int64 v30; // [rsp+8h] [rbp-48h]
  int v31; // [rsp+14h] [rbp-3Ch] BYREF
  const __m128i *v32[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = *(_BYTE *)(a1 + 56);
  if ( v3 == 103 || v3 == 88 )
  {
    v11 = *(const __m128i **)(a1 + 72);
    v29 = v11[1].m128i_i64[0];
    v6 = *(const __m128i **)(v29 + 16);
    v7 = (const __m128i *)sub_73F620(v11, a2);
    v12 = sub_7E6F30(v7, unk_4F06964, v32);
    v9 = (const __m128i *)v29;
    if ( v12 )
    {
      if ( v3 == 103 )
      {
        if ( !LODWORD(v32[0]) )
        {
          v13 = (const __m128i *)sub_73F620(v6, a2);
          sub_730620(a1, v13);
          return 1;
        }
LABEL_8:
        v10 = (const __m128i *)sub_73F620(v9, a2);
        sub_730620(a1, v10);
        return 1;
      }
      if ( !LODWORD(v32[0]) )
        goto LABEL_8;
LABEL_15:
      sub_730620(a1, v7);
      return 1;
    }
LABEL_16:
    v14 = sub_73F620(v9, a2);
    if ( v6 )
      v6 = (const __m128i *)sub_73F620(v6, a2);
    *(_QWORD *)(a1 + 72) = v7;
    v7[1].m128i_i64[0] = (__int64)v14;
    v14[2] = v6;
    return 1;
  }
  if ( v3 == 87 )
  {
    v5 = *(const __m128i **)(a1 + 72);
    v28 = v5[1].m128i_i64[0];
    v6 = *(const __m128i **)(v28 + 16);
    v7 = (const __m128i *)sub_73F620(v5, a2);
    v8 = sub_7E6F30(v7, unk_4F06964, v32);
    v9 = (const __m128i *)v28;
    if ( v8 )
    {
      if ( LODWORD(v32[0]) )
        goto LABEL_8;
      goto LABEL_15;
    }
    goto LABEL_16;
  }
  result = 0;
  if ( v3 == 73 && (*(_BYTE *)(a1 + 57) & 0xFB) == 2 )
  {
    v15 = *(const __m128i **)(a1 + 72);
    if ( v15[1].m128i_i8[8] == 3 )
    {
      v16 = v15[3].m128i_i64[1];
      if ( (*(_BYTE *)(v16 + 173) & 0x30) != 0 )
      {
        v17 = *(_QWORD *)(v16 + 264);
        if ( !v17 || (v18 = *(_DWORD *)(v17 + 16)) == 0 )
          BUG();
        v19 = (const __m128i *)v15[1].m128i_i64[0];
        if ( v18 != 1 )
        {
          v20 = *(_QWORD *)(v17 + 48);
          if ( v20 )
          {
            *(_DWORD *)(v17 + 16) = 1;
            *(_QWORD *)(v17 + 24) = v20;
            *(_QWORD *)(v17 + 48) = 0;
          }
        }
        v30 = v16;
        v21 = sub_73F620(v19, a2);
        if ( (unsigned int)sub_7E6B40(v21, 1, 1, 1, &v31)
          && ((*(_BYTE *)(v30 + 173) & 0x20) == 0 || v31)
          && !*(_BYTE *)(v17 + 57) )
        {
          v32[0] = (const __m128i *)sub_724DC0();
          v23 = v32[0];
          v24 = *(_QWORD *)(v17 + 24);
          *(_DWORD *)(v17 + 16) = 2;
          *(_QWORD *)(v17 + 48) = v24;
          *(_QWORD *)(v17 + 24) = v21;
          sub_72BB40(*(_QWORD *)a1, v23);
          sub_7264E0(a1, 2);
          *(_QWORD *)(a1 + 56) = sub_73A460(v32[0], 2, v25, v26, v27);
          sub_724E30((__int64)v32);
        }
        else
        {
          v22 = sub_73F620(v15, a2);
          v22[2] = v21;
          *(_QWORD *)(a1 + 72) = v22;
        }
        return 1;
      }
    }
  }
  return result;
}
