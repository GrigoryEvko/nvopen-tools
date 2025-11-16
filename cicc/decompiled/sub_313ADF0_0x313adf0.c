// Function: sub_313ADF0
// Address: 0x313adf0
//
__int64 __fastcall sub_313ADF0(__int64 a1, __int64 a2, __m128i *a3, int a4, char a5, char a6)
{
  char v10; // al
  int v12; // r9d
  __int64 *v13; // rsi
  __int64 v14; // rax
  _BYTE *v15; // rdx
  unsigned __int64 v16; // rsi
  __int64 v17; // rax
  __int16 v18; // dx
  __int64 v19; // rsi
  char v20; // al
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int64 v23; // rax
  int v24; // [rsp+8h] [rbp-88h]
  unsigned int v26; // [rsp+14h] [rbp-7Ch] BYREF
  unsigned __int64 v27; // [rsp+18h] [rbp-78h] BYREF
  _QWORD v28[2]; // [rsp+20h] [rbp-70h] BYREF
  _BYTE v29[16]; // [rsp+30h] [rbp-60h] BYREF
  void (__fastcall *v30)(_BYTE *, _BYTE *, __int64); // [rsp+40h] [rbp-50h]
  __int16 v31; // [rsp+50h] [rbp-40h]

  if ( sub_31387E0(a2, (__int64)a3) )
  {
    v12 = 192;
    if ( a4 != 66 )
    {
      v12 = 320;
      if ( a4 != 68 )
      {
        v12 = 64;
        if ( a4 == 5 )
          v12 = 32;
      }
    }
    v24 = v12;
    v13 = (__int64 *)sub_31376D0(a2, a3->m128i_i64, &v26);
    v28[0] = sub_313A9F0(a2, v13, v26, v24, 0);
    v14 = sub_313A9F0(a2, v13, v26, 0, 0);
    v28[1] = sub_3135D90(a2, v14);
    if ( !a5
      && (v21 = *(unsigned int *)(a2 + 8), (_DWORD)v21)
      && (v22 = *(_QWORD *)a2 + 40 * v21 - 40, *(_BYTE *)(v22 + 36))
      && *(_DWORD *)(v22 + 32) == 48 )
    {
      v31 = 257;
      v16 = 0;
      v15 = sub_3135910(a2, 2);
      if ( !v15 )
        goto LABEL_11;
    }
    else
    {
      a6 = 0;
      v31 = 257;
      v15 = sub_3135910(a2, 0);
      if ( !v15 )
      {
        sub_921880((unsigned int **)(a2 + 512), 0, 0, (int)v28, 2, (__int64)v29, 0);
LABEL_12:
        v18 = *(_WORD *)(a2 + 576);
        v19 = *(_QWORD *)(a2 + 560);
        v20 = *(_BYTE *)(a1 + 24) & 0xFC;
        *(_QWORD *)(a1 + 8) = *(_QWORD *)(a2 + 568);
        *(_QWORD *)a1 = v19;
        *(_BYTE *)(a1 + 24) = v20 | 2;
        *(_WORD *)(a1 + 16) = v18;
        return a1;
      }
    }
    v16 = *((_QWORD *)v15 + 3);
LABEL_11:
    v17 = sub_921880((unsigned int **)(a2 + 512), v16, (int)v15, (int)v28, 2, (__int64)v29, 0);
    if ( a6 )
    {
      v30 = 0;
      sub_3138420(&v27, (__int64 *)a2, v17, 48, (__int64)v29);
      if ( v30 )
        v30(v29, v29, 3);
      v23 = v27 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v27 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        *(_BYTE *)(a1 + 24) |= 3u;
        *(_QWORD *)a1 = v23;
        return a1;
      }
    }
    goto LABEL_12;
  }
  v10 = *(_BYTE *)(a1 + 24) & 0xFC;
  *(__m128i *)a1 = _mm_loadu_si128(a3);
  *(_BYTE *)(a1 + 24) = v10 | 2;
  *(_QWORD *)(a1 + 16) = a3[1].m128i_i64[0];
  return a1;
}
