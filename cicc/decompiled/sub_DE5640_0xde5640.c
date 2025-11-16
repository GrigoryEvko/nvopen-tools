// Function: sub_DE5640
// Address: 0xde5640
//
char __fastcall sub_DE5640(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v9; // rsi
  _QWORD *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned __int16 v19; // bx
  unsigned int v20; // r15d
  __int16 v21; // dx
  int v22; // eax
  __int64 *v23; // rax
  __int64 *v24; // rbx
  __int64 v25; // rcx
  __int64 v26; // r8
  unsigned __int64 v27; // rax
  int v28; // eax
  int v29; // edx
  __int64 v30; // rdi
  unsigned int v31; // ecx
  __int64 *v32; // rsi
  int v33; // eax
  int v34; // eax
  _QWORD *v35; // r8
  int v36; // eax
  int v37; // r9d
  _QWORD *v39; // [rsp+0h] [rbp-C0h]
  _QWORD *v40; // [rsp+0h] [rbp-C0h]
  _QWORD *v41; // [rsp+0h] [rbp-C0h]
  _QWORD *v42; // [rsp+0h] [rbp-C0h]
  unsigned __int16 v43; // [rsp+10h] [rbp-B0h]
  __int64 v45; // [rsp+20h] [rbp-A0h]
  int v46; // [rsp+2Ch] [rbp-94h] BYREF
  __int64 *v47; // [rsp+30h] [rbp-90h] BYREF
  unsigned __int64 v48; // [rsp+38h] [rbp-88h]
  _QWORD v49[2]; // [rsp+40h] [rbp-80h] BYREF
  _QWORD v50[2]; // [rsp+50h] [rbp-70h] BYREF
  _QWORD v51[12]; // [rsp+60h] [rbp-60h] BYREF

  v9 = *(_QWORD *)(a3 + 8);
  v46 = a6;
  LOBYTE(v10) = sub_D97040((__int64)a1, v9);
  if ( !(_BYTE)v10 )
    return (char)v10;
  v51[1] = a4;
  v51[2] = a5;
  v51[0] = a3;
  v51[3] = a1;
  v51[4] = &v46;
  v45 = sub_DE5430((__int64)v51, 0, v11, v12, v13, v14);
  v19 = v15;
  v43 = v15;
  LODWORD(v10) = *(_DWORD *)(a3 + 4) & 0x7FFFFFF;
  if ( (unsigned int)v10 > 1 )
  {
    v20 = 2;
    if ( !v45 )
      return (char)v10;
    while ( 1 )
    {
      while ( 1 )
      {
        v10 = (_QWORD *)sub_DE5430((__int64)v51, v20 - 1, v15, v16, v17, v18);
        if ( !v10 || v19 != v21 )
          return (char)v10;
        if ( v19 == 11 )
        {
          v42 = v10;
          v36 = sub_C49970(*(_QWORD *)(v45 + 32) + 24LL, (unsigned __int64 *)(v10[4] + 24LL));
          v16 = (__int64)v42;
          if ( v36 > 0 )
            v16 = v45;
          v45 = v16;
          if ( v20 >= (*(_DWORD *)(a3 + 4) & 0x7FFFFFFu) )
            goto LABEL_15;
          goto LABEL_23;
        }
        if ( v19 <= 0xBu )
          break;
        if ( v19 != 12 )
LABEL_45:
          BUG();
        v41 = v10;
        v34 = sub_C4C880(*(_QWORD *)(v45 + 32) + 24LL, v10[4] + 24LL);
        v16 = (__int64)v41;
        if ( v34 > 0 )
          v16 = v45;
        v45 = v16;
        if ( v20 >= (*(_DWORD *)(a3 + 4) & 0x7FFFFFFu) )
          goto LABEL_15;
LABEL_23:
        ++v20;
      }
      if ( v19 != 9 )
      {
        if ( v19 != 10 )
          goto LABEL_45;
        v40 = v10;
        v33 = sub_C4C880(*(_QWORD *)(v45 + 32) + 24LL, v10[4] + 24LL);
        v16 = (__int64)v40;
        if ( v33 < 0 )
          v16 = v45;
        v45 = v16;
        if ( v20 >= (*(_DWORD *)(a3 + 4) & 0x7FFFFFFu) )
          goto LABEL_15;
        goto LABEL_23;
      }
      v39 = v10;
      v22 = sub_C49970(*(_QWORD *)(v45 + 32) + 24LL, (unsigned __int64 *)(v10[4] + 24LL));
      v16 = (__int64)v39;
      if ( v22 < 0 )
        v16 = v45;
      v45 = v16;
      if ( v20 >= (*(_DWORD *)(a3 + 4) & 0x7FFFFFFu) )
        goto LABEL_15;
      ++v20;
    }
  }
  if ( !v45 )
    return (char)v10;
LABEL_15:
  v23 = sub_DD8400((__int64)a1, a3);
  v49[0] = v50;
  v24 = v23;
  v50[1] = v23;
  v50[0] = v45;
  v49[1] = 0x200000002LL;
  v27 = sub_DCD310(a1, v43, (__int64)v49, v25, v26);
  v47 = v24;
  v48 = v27;
  v28 = *(_DWORD *)(a2 + 24);
  if ( v28 )
  {
    v29 = v28 - 1;
    v30 = *(_QWORD *)(a2 + 8);
    v31 = (v28 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
    v10 = (_QWORD *)(v30 + 16LL * v31);
    v32 = (__int64 *)*v10;
    if ( v24 == (__int64 *)*v10 )
      goto LABEL_17;
    v37 = 1;
    v35 = 0;
    while ( v32 != (__int64 *)-4096LL )
    {
      if ( v32 != (__int64 *)-8192LL || v35 )
        v10 = v35;
      v31 = v29 & (v37 + v31);
      v32 = *(__int64 **)(v30 + 16LL * v31);
      if ( v24 == v32 )
        goto LABEL_17;
      ++v37;
      v35 = v10;
      v10 = (_QWORD *)(v30 + 16LL * v31);
    }
    if ( !v35 )
      v35 = v10;
  }
  else
  {
    v35 = 0;
  }
  v32 = (__int64 *)&v47;
  v10 = sub_DB06C0(a2, &v47, v35);
  *v10 = v47;
  v10[1] = v48;
LABEL_17:
  if ( (_QWORD *)v49[0] != v50 )
    LOBYTE(v10) = _libc_free(v49[0], v32);
  return (char)v10;
}
