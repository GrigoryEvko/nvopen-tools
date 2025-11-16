// Function: sub_34AB180
// Address: 0x34ab180
//
void __fastcall sub_34AB180(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r9
  unsigned int *v8; // r13
  unsigned int v9; // esi
  char v10; // al
  int v11; // edi
  __int64 v12; // rcx
  int v13; // edx
  __int64 v14; // r14
  __int64 v15; // r8
  const __m128i *v16; // r15
  __int64 v17; // rbx
  char v18; // al
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rdx
  int v22; // r8d
  __int64 v23; // rdx
  __int64 v24; // rdx
  unsigned __int64 v25; // rbx
  unsigned __int64 v26; // rdi
  int *v27; // [rsp+10h] [rbp-210h]
  char v29; // [rsp+2Fh] [rbp-1F1h]
  _BYTE *v30; // [rsp+30h] [rbp-1F0h] BYREF
  __int64 v31; // [rsp+38h] [rbp-1E8h]
  _BYTE v32[16]; // [rsp+40h] [rbp-1E0h] BYREF
  _BYTE *v33; // [rsp+50h] [rbp-1D0h] BYREF
  __int64 v34; // [rsp+58h] [rbp-1C8h]
  _BYTE v35[16]; // [rsp+60h] [rbp-1C0h] BYREF
  unsigned int *v36; // [rsp+70h] [rbp-1B0h] BYREF
  __int64 v37; // [rsp+78h] [rbp-1A8h]
  _BYTE v38[136]; // [rsp+80h] [rbp-1A0h] BYREF
  int v39; // [rsp+108h] [rbp-118h] BYREF
  unsigned __int64 v40; // [rsp+110h] [rbp-110h]
  int *v41; // [rsp+118h] [rbp-108h]
  int *v42; // [rsp+120h] [rbp-100h]
  __int64 v43; // [rsp+128h] [rbp-F8h]
  _DWORD *v44; // [rsp+130h] [rbp-F0h] BYREF
  int v45; // [rsp+138h] [rbp-E8h]
  int v46; // [rsp+13Ch] [rbp-E4h]
  _DWORD v47[34]; // [rsp+140h] [rbp-E0h] BYREF
  int v48; // [rsp+1C8h] [rbp-58h] BYREF
  unsigned __int64 v49; // [rsp+1D0h] [rbp-50h]
  int *v50; // [rsp+1D8h] [rbp-48h]
  int *v51; // [rsp+1E0h] [rbp-40h]
  __int64 v52; // [rsp+1E8h] [rbp-38h]

  v36 = (unsigned int *)v38;
  v37 = 0x2000000000LL;
  v44 = v47;
  v39 = 0;
  v40 = 0;
  v41 = &v39;
  v42 = &v39;
  v43 = 0;
  v46 = 32;
  v48 = 0;
  v49 = 0;
  v50 = &v48;
  v51 = &v48;
  v52 = 0;
  v47[0] = 0;
  v45 = 1;
  sub_34A6040((__int64)&v36, (__int64)&v44, a2, a3, a2, a6);
  if ( v43 )
  {
    v29 = 0;
    v8 = (unsigned int *)v41;
    v27 = &v39;
  }
  else
  {
    v8 = v36;
    v29 = 1;
    v27 = (int *)&v36[(unsigned int)v37];
  }
  if ( v29 )
    goto LABEL_23;
LABEL_4:
  if ( v27 == (int *)v8 )
    goto LABEL_38;
  v9 = v8[8];
  v10 = *(_BYTE *)(a3 + 56) & 1;
  if ( v10 )
  {
LABEL_6:
    v11 = *(_DWORD *)(a3 + 64);
    v12 = a3 + 64;
    v13 = 3;
    if ( v11 )
      goto LABEL_27;
    goto LABEL_7;
  }
  while ( 1 )
  {
    v21 = *(unsigned int *)(a3 + 72);
    v12 = *(_QWORD *)(a3 + 64);
    if ( !(_DWORD)v21 )
      goto LABEL_35;
    v11 = *(_DWORD *)v12;
    v13 = v21 - 1;
    if ( *(_DWORD *)v12 )
    {
LABEL_27:
      v22 = 1;
      v7 = 0;
      while ( v11 != -1 )
      {
        v7 = v13 & (unsigned int)(v22 + v7);
        v11 = *(_DWORD *)(v12 + 32LL * (unsigned int)v7);
        if ( !v11 )
        {
          v12 += 32LL * (unsigned int)v7;
          goto LABEL_7;
        }
        ++v22;
      }
      if ( v10 )
      {
        v23 = 128;
        goto LABEL_36;
      }
      v21 = *(unsigned int *)(a3 + 72);
LABEL_35:
      v23 = 32 * v21;
LABEL_36:
      v12 += v23;
    }
LABEL_7:
    v14 = *(_QWORD *)(a3 + 16);
    v15 = *(_QWORD *)(v12 + 8) + 384LL * v9;
    v16 = (const __m128i *)v15;
    if ( v14 )
    {
      v17 = a3 + 8;
      do
      {
        while ( 1 )
        {
          v18 = sub_34A0190(v14 + 32, (__int64)v16);
          v19 = *(_QWORD *)(v14 + 16);
          v12 = *(_QWORD *)(v14 + 24);
          if ( v18 )
            break;
          v17 = v14;
          v14 = *(_QWORD *)(v14 + 16);
          if ( !v19 )
            goto LABEL_12;
        }
        v14 = *(_QWORD *)(v14 + 24);
      }
      while ( v12 );
LABEL_12:
      if ( a3 + 8 != v17 && (unsigned __int8)sub_34A0190((__int64)v16, v17 + 32) )
        v17 = a3 + 8;
    }
    else
    {
      v17 = a3 + 8;
    }
    v20 = *(unsigned int *)(v17 + 424);
    v30 = v32;
    v31 = 0x200000000LL;
    if ( (_DWORD)v20 )
    {
      sub_349DD80((__int64)&v30, v17 + 416, v20, v12, v15, v7);
      v33 = v35;
      v34 = 0x200000000LL;
      if ( (_DWORD)v31 )
        sub_349DD80((__int64)&v33, (__int64)&v30, v24, v12, v15, (__int64)&v33);
    }
    else
    {
      v34 = 0x200000000LL;
      v33 = v35;
    }
    sub_34AADC0(a1, (__int64)&v33, v16, v12, v15, (__int64)&v33);
    if ( v33 != v35 )
      _libc_free((unsigned __int64)v33);
    if ( v30 != v32 )
      _libc_free((unsigned __int64)v30);
    if ( !v29 )
    {
      v8 = (unsigned int *)sub_220EF30((__int64)v8);
      goto LABEL_4;
    }
    ++v8;
LABEL_23:
    if ( v27 == (int *)v8 )
      break;
    v9 = *v8;
    v10 = *(_BYTE *)(a3 + 56) & 1;
    if ( v10 )
      goto LABEL_6;
  }
LABEL_38:
  sub_349E330(v49);
  if ( v44 != v47 )
    _libc_free((unsigned __int64)v44);
  v25 = v40;
  while ( v25 )
  {
    sub_349E500(*(_QWORD *)(v25 + 24));
    v26 = v25;
    v25 = *(_QWORD *)(v25 + 16);
    j_j___libc_free_0(v26);
  }
  if ( v36 != (unsigned int *)v38 )
    _libc_free((unsigned __int64)v36);
}
