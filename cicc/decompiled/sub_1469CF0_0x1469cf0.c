// Function: sub_1469CF0
// Address: 0x1469cf0
//
void __fastcall sub_1469CF0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        __int64 a6,
        unsigned __int64 *a7)
{
  char v8; // cl
  __int64 v9; // r8
  int v10; // esi
  unsigned int v11; // eax
  unsigned __int64 *v12; // r10
  unsigned __int64 v13; // r9
  unsigned int v14; // esi
  unsigned int v15; // eax
  unsigned __int64 *v16; // rdx
  int v17; // edi
  unsigned int v18; // r8d
  int v19; // r11d
  __int64 *v20; // r13
  unsigned __int64 *v21; // [rsp+8h] [rbp-98h] BYREF
  unsigned __int64 v22; // [rsp+10h] [rbp-90h] BYREF
  unsigned __int64 v23; // [rsp+18h] [rbp-88h]
  unsigned __int64 v24; // [rsp+20h] [rbp-80h]
  char v25; // [rsp+28h] [rbp-78h]
  char v26[8]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v27; // [rsp+38h] [rbp-68h]
  unsigned __int64 v28; // [rsp+40h] [rbp-60h]
  char v29[72]; // [rsp+58h] [rbp-48h] BYREF

  v22 = (4LL * a5) | a3 & 0xFFFFFFFFFFFFFFFBLL;
  v23 = *a7;
  v24 = a7[1];
  v25 = *((_BYTE *)a7 + 16);
  sub_16CCCB0(v26, v29, a7 + 3);
  v8 = *(_BYTE *)(a1 + 8) & 1;
  if ( v8 )
  {
    v9 = a1 + 16;
    v10 = 3;
  }
  else
  {
    v14 = *(_DWORD *)(a1 + 24);
    v9 = *(_QWORD *)(a1 + 16);
    if ( !v14 )
    {
      v15 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      v16 = 0;
      v17 = (v15 >> 1) + 1;
LABEL_10:
      v18 = 3 * v14;
      goto LABEL_11;
    }
    v10 = v14 - 1;
  }
  v11 = v10 & (v22 ^ (v22 >> 9));
  v12 = (unsigned __int64 *)(v9 + 104LL * v11);
  v13 = *v12;
  if ( v22 == *v12 )
    goto LABEL_4;
  v19 = 1;
  v16 = 0;
  while ( v13 != -4 )
  {
    if ( v13 != -16 || v16 )
      v12 = v16;
    v11 = v10 & (v19 + v11);
    v20 = (__int64 *)(v9 + 104LL * v11);
    v13 = *v20;
    if ( v22 == *v20 )
      goto LABEL_4;
    v16 = v12;
    ++v19;
    v12 = (unsigned __int64 *)(v9 + 104LL * v11);
  }
  v15 = *(_DWORD *)(a1 + 8);
  if ( !v16 )
    v16 = v12;
  ++*(_QWORD *)a1;
  v17 = (v15 >> 1) + 1;
  if ( !v8 )
  {
    v14 = *(_DWORD *)(a1 + 24);
    goto LABEL_10;
  }
  v18 = 12;
  v14 = 4;
LABEL_11:
  if ( 4 * v17 >= v18 )
  {
    v14 *= 2;
    goto LABEL_23;
  }
  if ( v14 - *(_DWORD *)(a1 + 12) - v17 <= v14 >> 3 )
  {
LABEL_23:
    sub_1469840(a1, v14);
    sub_145FDD0(a1, (__int64 *)&v22, &v21);
    v16 = v21;
    v15 = *(_DWORD *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 8) = (2 * (v15 >> 1) + 2) | v15 & 1;
  if ( *v16 != -4 )
    --*(_DWORD *)(a1 + 12);
  *v16 = v22;
  v16[1] = v23;
  v16[2] = v24;
  *((_BYTE *)v16 + 24) = v25;
  sub_16CCEE0(v16 + 4, v16 + 9, 4, v26);
LABEL_4:
  if ( v28 != v27 )
    _libc_free(v28);
}
