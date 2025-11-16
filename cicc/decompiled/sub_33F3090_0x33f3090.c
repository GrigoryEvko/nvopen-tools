// Function: sub_33F3090
// Address: 0x33f3090
//
unsigned __int64 __fastcall sub_33F3090(_QWORD *a1, __int64 a2, unsigned __int64 a3, __int64 a4, char a5)
{
  unsigned __int64 v5; // r12
  unsigned __int16 *v9; // rax
  __m128i *v10; // rax
  int v11; // edx
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // r8
  unsigned __int64 v15; // rax
  __int64 v16; // rdx
  _QWORD *v17; // r8
  unsigned __int64 v18; // r8
  int v19; // r9d
  __int64 v20; // rsi
  unsigned __int8 *v21; // rsi
  __int64 v22; // rcx
  unsigned __int64 v23; // rax
  __int64 v24; // rax
  unsigned __int64 v25; // [rsp+0h] [rbp-110h]
  int v26; // [rsp+0h] [rbp-110h]
  int v27; // [rsp+8h] [rbp-108h]
  __m128i *v28; // [rsp+10h] [rbp-100h]
  int v29; // [rsp+20h] [rbp-F0h]
  unsigned __int64 v30; // [rsp+20h] [rbp-F0h]
  __int64 *v31; // [rsp+20h] [rbp-F0h]
  int v32; // [rsp+20h] [rbp-F0h]
  __int64 *v33; // [rsp+38h] [rbp-D8h] BYREF
  unsigned __int8 *v34; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v35; // [rsp+48h] [rbp-C8h]
  _BYTE *v36; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v37; // [rsp+58h] [rbp-B8h]
  _BYTE v38[176]; // [rsp+60h] [rbp-B0h] BYREF

  v5 = a3;
  if ( !a5 )
    return v5;
  v9 = (unsigned __int16 *)(*(_QWORD *)(a3 + 48) + 16LL * (unsigned int)a4);
  v10 = sub_33ED250((__int64)a1, *v9, *((_QWORD *)v9 + 1));
  v37 = 0x2000000000LL;
  v27 = v11;
  v36 = v38;
  v28 = v10;
  v34 = (unsigned __int8 *)v5;
  v35 = a4;
  sub_33C9670((__int64)&v36, 5, (unsigned __int64)v10, (unsigned __int64 *)&v34, 1, (__int64)&v36);
  v12 = (unsigned int)v37;
  v13 = 1LL << a5;
  v14 = (unsigned int)v37 + 1LL;
  if ( v14 > HIDWORD(v37) )
  {
    sub_C8D5F0((__int64)&v36, v38, (unsigned int)v37 + 1LL, 4u, v14, (__int64)&v36);
    v12 = (unsigned int)v37;
    v13 = 1LL << a5;
  }
  *(_DWORD *)&v36[4 * v12] = v13;
  v15 = HIDWORD(v13);
  LODWORD(v37) = v37 + 1;
  v16 = (unsigned int)v37;
  if ( (unsigned __int64)(unsigned int)v37 + 1 > HIDWORD(v37) )
  {
    v26 = v15;
    sub_C8D5F0((__int64)&v36, v38, (unsigned int)v37 + 1LL, 4u, (unsigned int)v37 + 1LL, (__int64)&v36);
    v16 = (unsigned int)v37;
    LODWORD(v15) = v26;
  }
  *(_DWORD *)&v36[4 * v16] = v15;
  LODWORD(v37) = v37 + 1;
  v33 = 0;
  v17 = sub_33CCCF0((__int64)a1, (__int64)&v36, a2, (__int64 *)&v33);
  if ( !v17 )
  {
    v18 = a1[52];
    v19 = *(_DWORD *)(a2 + 8);
    if ( v18 )
    {
      a1[52] = *(_QWORD *)v18;
    }
    else
    {
      v22 = a1[53];
      a1[63] += 120LL;
      v23 = (v22 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      if ( a1[54] >= v23 + 120 && v22 )
      {
        a1[53] = v23 + 120;
        if ( !v23 )
        {
LABEL_17:
          v31 = (__int64 *)v18;
          v34 = (unsigned __int8 *)v5;
          v35 = a4;
          sub_33E4EC0((__int64)a1, v18, (__int64)&v34, 1);
          sub_C657C0(a1 + 65, v31, v33, (__int64)off_4A367D0);
          sub_33CC420((__int64)a1, (__int64)v31);
          v17 = v31;
          goto LABEL_8;
        }
        v18 = (v22 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      }
      else
      {
        v32 = v19;
        v24 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
        v19 = v32;
        v18 = v24;
      }
    }
    v20 = *(_QWORD *)a2;
    v34 = (unsigned __int8 *)v20;
    if ( v20 )
    {
      v25 = v18;
      v29 = v19;
      sub_B96E90((__int64)&v34, v20, 1);
      v18 = v25;
      v19 = v29;
    }
    *(_QWORD *)v18 = 0;
    *(_QWORD *)(v18 + 8) = 0;
    *(_QWORD *)(v18 + 48) = v28;
    *(_QWORD *)(v18 + 16) = 0;
    *(_QWORD *)(v18 + 24) = 5;
    *(_WORD *)(v18 + 34) = -1;
    *(_DWORD *)(v18 + 36) = -1;
    *(_QWORD *)(v18 + 40) = 0;
    *(_QWORD *)(v18 + 56) = 0;
    *(_DWORD *)(v18 + 64) = 0;
    *(_DWORD *)(v18 + 68) = v27;
    *(_DWORD *)(v18 + 72) = v19;
    v21 = v34;
    *(_QWORD *)(v18 + 80) = v34;
    if ( v21 )
    {
      v30 = v18;
      sub_B976B0((__int64)&v34, v21, v18 + 80);
      v18 = v30;
      *(_QWORD *)(v30 + 88) = 0xFFFFFFFFLL;
      *(_WORD *)(v30 + 32) = 0;
    }
    else
    {
      *(_QWORD *)(v18 + 88) = 0xFFFFFFFFLL;
      *(_WORD *)(v18 + 32) = 0;
    }
    *(_BYTE *)(v18 + 96) = a5;
    goto LABEL_17;
  }
LABEL_8:
  v5 = (unsigned __int64)v17;
  if ( v36 != v38 )
    _libc_free((unsigned __int64)v36);
  return v5;
}
