// Function: sub_3267C60
// Address: 0x3267c60
//
__int64 __fastcall sub_3267C60(_QWORD *a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 *v5; // rdx
  unsigned int v6; // r12d
  __int64 v7; // r13
  int v8; // eax
  unsigned __int64 *v9; // rax
  __int64 v10; // rdi
  bool v11; // cf
  __int64 v12; // r13
  __int64 v14; // rax
  unsigned int v17; // r15d
  int v18; // r9d
  unsigned int v19; // r10d
  unsigned int v22; // eax
  unsigned __int16 *v23; // r15
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  unsigned int v26; // edx
  __int64 v27; // rax
  unsigned int v28; // edx
  unsigned int v29; // [rsp+8h] [rbp-B8h]
  unsigned int v30; // [rsp+8h] [rbp-B8h]
  unsigned int v31; // [rsp+8h] [rbp-B8h]
  unsigned int v32; // [rsp+Ch] [rbp-B4h]
  __int128 v33; // [rsp+10h] [rbp-B0h]
  __int128 v34; // [rsp+20h] [rbp-A0h]
  unsigned __int8 v35; // [rsp+5Fh] [rbp-61h] BYREF
  __int64 v36; // [rsp+60h] [rbp-60h] BYREF
  int v37; // [rsp+68h] [rbp-58h]
  unsigned __int64 v38; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v39; // [rsp+78h] [rbp-48h]
  unsigned __int64 v40; // [rsp+80h] [rbp-40h]
  unsigned int v41; // [rsp+88h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 80);
  v36 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v36, v4, 1);
  v5 = *(__int64 **)(a2 + 40);
  v6 = *(unsigned __int8 *)(a2 + 96);
  v37 = *(_DWORD *)(a2 + 72);
  v7 = *v5;
  v35 = v6;
  v8 = *(_DWORD *)(v7 + 24);
  if ( v8 == 5 )
  {
    v9 = (unsigned __int64 *)&v35;
    v10 = *a1;
    v11 = (unsigned __int8)v6 < *(_BYTE *)(v7 + 96);
    LOBYTE(v38) = *(_BYTE *)(v7 + 96);
    if ( v11 )
      v9 = &v38;
    v12 = sub_33F3090(
            v10,
            &v36,
            **(_QWORD **)(v7 + 40),
            *(_QWORD *)(*(_QWORD *)(v7 + 40) + 8LL),
            *(unsigned __int8 *)v9);
    goto LABEL_7;
  }
  if ( (unsigned int)(v8 - 56) > 1 )
  {
LABEL_11:
    v12 = 0;
    goto LABEL_7;
  }
  v32 = *((_DWORD *)v5 + 2);
  v14 = *(_QWORD *)(v7 + 40);
  v34 = (__int128)_mm_loadu_si128((const __m128i *)v14);
  v33 = (__int128)_mm_loadu_si128((const __m128i *)(v14 + 40));
  sub_33DD090(&v38, *a1, v34, *((_QWORD *)&v34 + 1), 0);
  if ( v39 > 0x40 )
  {
    v17 = sub_C445E0((__int64)&v38);
    if ( v41 <= 0x40 || (v25 = v40) == 0 )
    {
LABEL_36:
      if ( v38 )
        j_j___libc_free_0_0(v38);
      goto LABEL_16;
    }
LABEL_35:
    j_j___libc_free_0_0(v25);
    if ( v39 <= 0x40 )
      goto LABEL_16;
    goto LABEL_36;
  }
  _RAX = ~v38;
  __asm { tzcnt   rdx, rax }
  v17 = _RDX;
  if ( v38 == -1 )
    v17 = 64;
  if ( v41 > 0x40 )
  {
    v25 = v40;
    if ( v40 )
      goto LABEL_35;
  }
LABEL_16:
  sub_33DD090(&v38, *a1, v33, *((_QWORD *)&v33 + 1), 0);
  if ( v39 > 0x40 )
  {
    v19 = sub_C445E0((__int64)&v38);
    if ( v41 <= 0x40 || (v24 = v40) == 0 )
    {
LABEL_31:
      if ( v38 )
      {
        v30 = v19;
        j_j___libc_free_0_0(v38);
        v19 = v30;
      }
      goto LABEL_20;
    }
LABEL_30:
    v29 = v19;
    j_j___libc_free_0_0(v24);
    v19 = v29;
    if ( v39 <= 0x40 )
      goto LABEL_20;
    goto LABEL_31;
  }
  v19 = 64;
  _RAX = ~v38;
  __asm { tzcnt   rdx, rax }
  if ( v38 != -1 )
    v19 = _RDX;
  if ( v41 > 0x40 )
  {
    v24 = v40;
    if ( v40 )
      goto LABEL_30;
  }
LABEL_20:
  v22 = v19;
  if ( v17 >= v19 )
    v22 = v17;
  if ( v22 < v6 )
    goto LABEL_11;
  if ( v6 > v17 )
  {
    v31 = v19;
    v27 = sub_33F3090(*a1, &v36, v34, *((_QWORD *)&v34 + 1), v35);
    v19 = v31;
    *(_QWORD *)&v34 = v27;
    *((_QWORD *)&v34 + 1) = v28 | *((_QWORD *)&v34 + 1) & 0xFFFFFFFF00000000LL;
  }
  if ( v6 > v19 )
  {
    *(_QWORD *)&v33 = sub_33F3090(*a1, &v36, v33, *((_QWORD *)&v33 + 1), v35);
    *((_QWORD *)&v33 + 1) = v26 | *((_QWORD *)&v33 + 1) & 0xFFFFFFFF00000000LL;
  }
  v23 = (unsigned __int16 *)(*(_QWORD *)(v7 + 48) + 16LL * v32);
  v12 = sub_3406EB0(*a1, *(_DWORD *)(v7 + 24), (unsigned int)&v36, *v23, *((_QWORD *)v23 + 1), v18, v34, v33);
LABEL_7:
  if ( v36 )
    sub_B91220((__int64)&v36, v36);
  return v12;
}
