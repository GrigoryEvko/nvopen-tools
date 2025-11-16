// Function: sub_325D1E0
// Address: 0x325d1e0
//
void __fastcall sub_325D1E0(__int64 a1, __int64 a2)
{
  int v4; // r14d
  __int64 v5; // r13
  __int64 v6; // rcx
  unsigned __int64 v7; // rax
  int v8; // edx
  __int64 v9; // rsi
  int v10; // edi
  __int64 v11; // r9
  __int64 v12; // r8
  unsigned __int64 v13; // r8
  unsigned __int64 v14; // rcx
  const __m128i *v15; // rdx
  __m128i *v16; // rax
  unsigned __int64 v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdx
  __int64 v22; // rcx
  unsigned __int64 v23; // r8
  __int64 v24; // r9
  _BYTE *v25; // rcx
  _QWORD *v26; // rdx
  unsigned __int8 *v27; // rax
  unsigned __int8 *v28; // rax
  __int64 v29; // rdi
  __int64 v30; // rdi
  __int64 v31; // r9
  char *v32; // r15
  __int64 v33; // [rsp+0h] [rbp-50h] BYREF
  int v34; // [rsp+8h] [rbp-48h]
  __int64 v35; // [rsp+10h] [rbp-40h]
  int v36; // [rsp+18h] [rbp-38h]

  if ( *(_BYTE *)(a1 + 24) || *(_BYTE *)(a1 + 26) || *(_BYTE *)(a1 + 25) )
  {
    if ( (*(_BYTE *)(*(_QWORD *)a2 + 2LL) & 8) != 0 )
    {
      v27 = (unsigned __int8 *)sub_B2E500(*(_QWORD *)a2);
      v28 = sub_BD3990(v27, a2);
      v4 = sub_B2A630((__int64)v28);
      sub_3259990(a1, a2);
      if ( v4 == 8 && *(_BYTE *)(a2 + 580) )
        return;
    }
    else
    {
      v4 = 0;
      sub_3259990(a1, a2);
    }
    if ( *(_BYTE *)(a1 + 24) || *(_BYTE *)(a1 + 25) )
    {
      v5 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL);
      v6 = *(unsigned int *)(v5 + 128);
      v7 = *(_QWORD *)(v5 + 120);
      v8 = *(_DWORD *)(v5 + 128);
      v9 = 32 * v6;
      if ( (_DWORD)v6 )
      {
        v29 = v7 + v9 - 32;
        v12 = *(_QWORD *)(v29 + 16);
        v8 = *(_DWORD *)(v29 + 24);
        v11 = *(_QWORD *)v29;
        v10 = *(_DWORD *)(v29 + 8);
      }
      else
      {
        v10 = 0;
        v11 = 0;
        v12 = 0;
      }
      v35 = v12;
      v13 = v6 + 1;
      v14 = *(unsigned int *)(v5 + 132);
      v36 = v8;
      v15 = (const __m128i *)&v33;
      v33 = v11;
      v34 = v10;
      if ( v13 > v14 )
      {
        v30 = v5 + 120;
        v31 = v5 + 136;
        if ( v7 > (unsigned __int64)&v33 || (unsigned __int64)&v33 >= v7 + v9 )
        {
          sub_C8D5F0(v30, (const void *)(v5 + 136), v13, 0x20u, v13, v31);
          v7 = *(_QWORD *)(v5 + 120);
          v15 = (const __m128i *)&v33;
          v9 = 32LL * *(unsigned int *)(v5 + 128);
        }
        else
        {
          v32 = (char *)&v33 - v7;
          sub_C8D5F0(v30, (const void *)(v5 + 136), v13, 0x20u, v13, v31);
          v7 = *(_QWORD *)(v5 + 120);
          v15 = (const __m128i *)&v32[v7];
          v9 = 32LL * *(unsigned int *)(v5 + 128);
        }
      }
      v16 = (__m128i *)(v9 + v7);
      *v16 = _mm_loadu_si128(v15);
      v16[1] = _mm_loadu_si128(v15 + 1);
      ++*(_DWORD *)(v5 + 128);
      v17 = sub_E99A60(
              *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
              *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL) + 288LL) + 8LL));
      (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD, __int64, __int64, __int64, __int64, int, __int64, int))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 176LL))(
        *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
        v17,
        0,
        v18,
        v19,
        v20,
        v33,
        v34,
        v35,
        v36);
      switch ( v4 )
      {
        case 8:
          sub_3259570(a1, (__int64 *)a2, v21, v22, v23);
          break;
        case 7:
          sub_325B8C0(a1, a2);
          break;
        case 9:
          sub_325A130(a1, a2);
          break;
        case 10:
          sub_325BEE0(a1, (_QWORD *)a2);
          break;
        default:
          sub_32530C0((char *)a1, v17, v21, v22, v23, v24);
          break;
      }
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 168LL))(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL));
    }
    v25 = *(_BYTE **)(a2 + 416);
    v26 = *(_QWORD **)(a2 + 408);
    if ( v25 != (_BYTE *)v26 )
      sub_3258570(a1 + 48, *(_BYTE **)(a1 + 56), v26, v25);
  }
}
