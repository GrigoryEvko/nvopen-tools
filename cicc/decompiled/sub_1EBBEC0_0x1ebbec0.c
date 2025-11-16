// Function: sub_1EBBEC0
// Address: 0x1ebbec0
//
__int64 __fastcall sub_1EBBEC0(_QWORD *a1, __int64 a2, int a3)
{
  unsigned __int16 *v3; // rbx
  __int64 v4; // r9
  unsigned __int16 *v5; // r12
  int v6; // r10d
  __int64 v7; // r11
  unsigned int v8; // r13d
  unsigned __int16 *v9; // rsi
  __int64 v11; // rdx
  unsigned int v12; // eax
  __int16 v13; // bx
  _WORD *v14; // rax
  __int16 *v15; // rcx
  unsigned __int16 v16; // bx
  __int16 *v17; // r13
  __int64 v18; // rax
  __int16 v19; // ax
  unsigned int v21; // [rsp+Ch] [rbp-144h]
  unsigned __int16 *v23; // [rsp+20h] [rbp-130h] BYREF
  unsigned int v24; // [rsp+28h] [rbp-128h]
  char v25; // [rsp+30h] [rbp-120h] BYREF
  __int64 v26; // [rsp+50h] [rbp-100h]
  int v27; // [rsp+58h] [rbp-F8h]
  int v28; // [rsp+60h] [rbp-F0h]
  char v29; // [rsp+64h] [rbp-ECh]
  _QWORD v30[4]; // [rsp+70h] [rbp-E0h] BYREF
  _BYTE *v31; // [rsp+90h] [rbp-C0h]
  __int64 v32; // [rsp+98h] [rbp-B8h]
  _BYTE v33[64]; // [rsp+A0h] [rbp-B0h] BYREF
  _BYTE *v34; // [rsp+E0h] [rbp-70h]
  __int64 v35; // [rsp+E8h] [rbp-68h]
  _BYTE v36[32]; // [rsp+F0h] [rbp-60h] BYREF
  __int16 v37; // [rsp+110h] [rbp-40h]
  __int64 v38; // [rsp+114h] [rbp-3Ch]

  sub_20C72B0(&v23, *(unsigned int *)(a2 + 112), a1[32], a1 + 35, a1[34]);
  v3 = v23;
  while ( 1 )
  {
    v4 = v28;
    v5 = v3;
    if ( v28 >= 0 )
      break;
    ++v28;
    v21 = v3[v24 + v4];
LABEL_12:
    if ( !v21 )
      goto LABEL_8;
    if ( a3 != v21 )
    {
      v11 = a1[87];
      if ( !v11 )
        BUG();
      v12 = *(_DWORD *)(*(_QWORD *)(v11 + 8) + 24LL * v21 + 16);
      v13 = v21 * (v12 & 0xF);
      v14 = (_WORD *)(*(_QWORD *)(v11 + 56) + 2LL * (v12 >> 4));
      v15 = v14 + 1;
      v16 = *v14 + v13;
LABEL_16:
      v17 = v15;
      if ( !v15 )
      {
LABEL_24:
        v5 = v23;
        goto LABEL_8;
      }
      while ( 1 )
      {
        v18 = *(_QWORD *)(a1[34] + 384LL);
        v30[3] = 0;
        v34 = v36;
        v38 = 0;
        v30[0] = v18 + 216LL * v16;
        v30[1] = a2;
        v31 = v33;
        v32 = 0x400000000LL;
        v35 = 0x400000000LL;
        v37 = 0;
        if ( (unsigned int)sub_20FD0B0(v30, 1) )
          break;
        if ( v34 != v36 )
          _libc_free((unsigned __int64)v34);
        if ( v31 != v33 )
          _libc_free((unsigned __int64)v31);
        v19 = *v17;
        v15 = 0;
        ++v17;
        if ( !v19 )
          goto LABEL_16;
        v16 += v19;
        if ( !v17 )
          goto LABEL_24;
      }
      if ( v34 != v36 )
        _libc_free((unsigned __int64)v34);
      if ( v31 != v33 )
        _libc_free((unsigned __int64)v31);
      v3 = v23;
    }
  }
  if ( !v29 )
  {
    v6 = v27;
    v7 = 2LL * v28;
    while ( v6 > (int)v4 )
    {
      v28 = v4 + 1;
      v8 = *(unsigned __int16 *)(v26 + v7);
      v9 = &v3[v24];
      LODWORD(v30[0]) = v8;
      if ( v9 == sub_1EBB4B0(v3, (__int64)v9, (int *)v30) )
      {
        v21 = v8;
        goto LABEL_12;
      }
    }
  }
  v21 = 0;
LABEL_8:
  if ( v5 != (unsigned __int16 *)&v25 )
    _libc_free((unsigned __int64)v5);
  return v21;
}
