// Function: sub_806C20
// Address: 0x806c20
//
_DWORD *__fastcall sub_806C20(__int64 *a1)
{
  __m128i *v1; // r14
  __int64 v2; // rax
  __int64 v3; // r15
  __int16 v4; // dx
  __int64 v5; // r13
  __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 v11; // rdi
  _QWORD *v12; // rbx
  __m128i *v13; // rax
  _BYTE *v14; // rax
  __int64 v15; // r8
  __int64 v16; // r9
  __int16 v17; // dx
  __int64 *v18; // rax
  _BYTE *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v25; // [rsp+8h] [rbp-A8h]
  __m128i *v26; // [rsp+8h] [rbp-A8h]
  __int64 v27; // [rsp+10h] [rbp-A0h]
  int v28; // [rsp+10h] [rbp-A0h]
  bool v29; // [rsp+1Bh] [rbp-95h]
  int v30; // [rsp+1Ch] [rbp-94h]
  __int16 v31; // [rsp+20h] [rbp-90h]
  __int16 v32; // [rsp+22h] [rbp-8Eh]
  int v33; // [rsp+24h] [rbp-8Ch]
  __int64 *v34; // [rsp+28h] [rbp-88h]
  __m128i *v35; // [rsp+30h] [rbp-80h] BYREF
  __m128i *v36; // [rsp+38h] [rbp-78h] BYREF
  __m128i v37[2]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v38[2]; // [rsp+60h] [rbp-50h] BYREF
  __int64 v39; // [rsp+70h] [rbp-40h]
  __int64 v40; // [rsp+78h] [rbp-38h]

  v1 = (__m128i *)a1[10];
  v2 = a1[4];
  v3 = a1[5];
  v4 = v1->m128i_i16[2];
  v33 = dword_4D03F38[0];
  v32 = dword_4D03F38[1];
  v30 = dword_4F07508[0];
  v31 = dword_4F07508[1];
  dword_4D03F38[0] = v1->m128i_i32[0];
  LOWORD(dword_4D03F38[1]) = v4;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)dword_4D03F38;
  v5 = *(_QWORD *)(*(_QWORD *)(v2 + 40) + 32LL);
  *(_BYTE *)(v5 + 88) |= 4u;
  v25 = v2;
  v34 = *(__int64 **)(v5 + 168);
  if ( v1[2].m128i_i8[8] == 19 )
  {
    sub_7E7150(v1, (__int64)v37, &v35);
    v29 = 0;
    v8 = v25;
    v28 = 1;
  }
  else
  {
    v6 = a1[6];
    v35 = (__m128i *)v1[4].m128i_i64[1];
    v27 = v6;
    sub_7E1740((__int64)v1, (__int64)v37);
    v7 = v27;
    v28 = 0;
    v8 = v25;
    v29 = v7 != 0;
  }
  if ( (*(_BYTE *)(v5 + 176) & 0x10) != 0 && (*(_BYTE *)(v8 + 205) & 0x1C) == 8 )
    sub_7FDF40(a1[4], 1, 1);
  sub_7E91D0(a1[11], (__int64)v37);
  v38[0] = 0;
  v38[1] = 0;
  v9 = v34[28];
  v39 = 0;
  v40 = 0;
  if ( v9 )
  {
    v9 = *(_QWORD *)(v3 + 112);
    v39 = v9;
  }
  sub_7F5B50(v5, v3, v9, 0, v37[0].m128i_i32);
  if ( *v34 )
  {
    v26 = v1;
    v10 = *v34;
    while ( 1 )
    {
      v17 = *(_WORD *)(v10 + 136);
      if ( v17 )
        break;
      if ( *(_QWORD *)(v10 + 128) != -1 )
      {
        v11 = v34[24];
        if ( v11 )
        {
          v12 = sub_7F5690(v11, v5, v10);
          goto LABEL_13;
        }
      }
LABEL_15:
      v10 = *(_QWORD *)v10;
      if ( !v10 )
      {
        v1 = v26;
        goto LABEL_19;
      }
    }
    v18 = sub_7FCC60(v39, 0, v17);
    v19 = sub_73DCD0(v18);
    v12 = sub_731370((__int64)v19, 0, v20, v21, v22, v23);
LABEL_13:
    if ( v12 )
    {
      v13 = sub_7E8890(v3, v10, 0);
      v14 = sub_7E45A0(v13->m128i_i64);
      sub_7E6A80(v14, 0x49u, (__int64)v12, v37[0].m128i_i32, v15, v16);
    }
    goto LABEL_15;
  }
LABEL_19:
  if ( v29 )
    sub_8062F0(v38);
  if ( v28 )
  {
    sub_7DE0F0((__int64)v35, 1u, (__int64)v38);
    sub_7E1720((__int64)v35, (__int64)v37);
    sub_806A20(v37, (__int64)v1, &v36);
  }
  else
  {
    sub_7EDD70(v35, &v36);
    sub_7E1720((__int64)v36, (__int64)v37);
    sub_806BE0(v37, (__int64)v1, v38);
  }
  dword_4F07508[0] = v30;
  LOWORD(dword_4F07508[1]) = v31;
  dword_4D03F38[0] = v33;
  LOWORD(dword_4D03F38[1]) = v32;
  return dword_4D03F38;
}
