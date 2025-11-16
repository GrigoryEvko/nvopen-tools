// Function: sub_2E8F280
// Address: 0x2e8f280
//
__int64 __fastcall sub_2E8F280(__int64 a1, unsigned int a2, _QWORD *a3, int a4)
{
  int v4; // r11d
  bool v5; // r9
  __int32 v6; // r14d
  __int64 v7; // r12
  int v8; // ebx
  int v9; // r13d
  int v10; // r13d
  __int64 v11; // rbx
  int v12; // r8d
  __int64 v14; // rax
  unsigned int v15; // ecx
  __int64 v16; // rax
  unsigned int v17; // r15d
  __int64 v18; // rbx
  __int64 v19; // rax
  bool v20; // zf
  __int64 v21; // r12
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // r10
  __int16 *v26; // rax
  int v27; // esi
  __int64 v28; // rax
  __int16 v29; // di
  __int32 v30; // esi
  __int16 *v31; // rax
  int v32; // r10d
  __int64 v33; // rax
  bool v34; // al
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  int v37; // [rsp+8h] [rbp-D8h]
  unsigned __int8 v38; // [rsp+Eh] [rbp-D2h]
  bool v39; // [rsp+Fh] [rbp-D1h]
  __int64 v40; // [rsp+10h] [rbp-D0h]
  __int64 v41; // [rsp+18h] [rbp-C8h]
  __int64 v42; // [rsp+30h] [rbp-B0h]
  unsigned int v44; // [rsp+40h] [rbp-A0h]
  unsigned int v45; // [rsp+44h] [rbp-9Ch]
  unsigned __int8 v46; // [rsp+44h] [rbp-9Ch]
  unsigned int v48; // [rsp+5Ch] [rbp-84h] BYREF
  _BYTE *v49; // [rsp+60h] [rbp-80h] BYREF
  __int64 v50; // [rsp+68h] [rbp-78h]
  _BYTE v51[16]; // [rsp+70h] [rbp-70h] BYREF
  __m128i v52; // [rsp+80h] [rbp-60h] BYREF
  __int64 v53; // [rsp+90h] [rbp-50h]
  __int64 v54; // [rsp+98h] [rbp-48h]
  __int64 v55; // [rsp+A0h] [rbp-40h]
  __int16 v56; // [rsp+A8h] [rbp-38h]

  v4 = a4;
  v5 = 0;
  v6 = a2;
  v7 = a1;
  v8 = a4;
  v45 = a2 - 1;
  if ( a2 - 1 <= 0x3FFFFFFE )
  {
    sub_E922F0(a3, a2);
    v4 = a4;
    v5 = 2 * v23 != 2;
  }
  v9 = *(_DWORD *)(a1 + 40);
  v49 = v51;
  v50 = 0x400000000LL;
  v10 = v9 & 0xFFFFFF;
  if ( !v10 )
  {
    v17 = 0;
    goto LABEL_28;
  }
  v11 = 0;
  v12 = 0;
  v42 = 24LL * a2;
  do
  {
    while ( 1 )
    {
      v14 = *(_QWORD *)(v7 + 32) + 40 * v11;
      if ( *(_BYTE *)v14 )
        goto LABEL_8;
      if ( (*(_BYTE *)(v14 + 3) & 0x10) != 0 )
        goto LABEL_8;
      if ( (*(_BYTE *)(v14 + 4) & 1) != 0 )
        goto LABEL_8;
      if ( (*(_BYTE *)(v14 + 4) & 8) != 0 )
        goto LABEL_8;
      v15 = *(_DWORD *)(v14 + 8);
      if ( !v15 )
        goto LABEL_8;
      if ( a2 == v15 )
        break;
      if ( v5 && (*(_BYTE *)(v14 + 3) & 0x40) != 0 && v15 - 1 <= 0x3FFFFFFE )
      {
        v37 = v4;
        v38 = v12;
        v24 = a3[7];
        v25 = a3[1];
        v39 = v5;
        v48 = *(_DWORD *)(v14 + 8);
        v44 = v15;
        v40 = v25;
        v41 = v24;
        v26 = (__int16 *)(v24 + 2LL * *(unsigned int *)(v25 + v42 + 8));
        v27 = *v26;
        v28 = (__int64)(v26 + 1);
        LODWORD(v54) = 0;
        v55 = 0;
        v29 = v27;
        v30 = a2 + v27;
        v52.m128i_i32[0] = v30;
        if ( !v29 )
          v28 = 0;
        LOWORD(v53) = v30;
        v52.m128i_i64[1] = v28;
        v56 = 0;
        if ( sub_2E46590(v52.m128i_i32, (int *)&v48) )
        {
LABEL_43:
          v17 = 1;
          goto LABEL_30;
        }
        v48 = a2;
        v31 = (__int16 *)(v41 + 2LL * *(unsigned int *)(v40 + 24LL * v44 + 8));
        v32 = *v31;
        v33 = (__int64)(v31 + 1);
        if ( !(_WORD)v32 )
          v33 = 0;
        v52.m128i_i32[0] = v32 + v44;
        LOWORD(v53) = v32 + v44;
        v52.m128i_i64[1] = v33;
        v34 = sub_2E46590(v52.m128i_i32, (int *)&v48);
        v5 = v39;
        v12 = v38;
        v4 = v37;
        if ( v34 )
        {
          v35 = (unsigned int)v50;
          v36 = (unsigned int)v50 + 1LL;
          if ( v36 > HIDWORD(v50) )
          {
            sub_C8D5F0((__int64)&v49, v51, v36, 4u, v38, v39);
            v35 = (unsigned int)v50;
            v4 = v37;
            v12 = v38;
            v5 = v39;
          }
          *(_DWORD *)&v49[4 * v35] = v11;
          LODWORD(v50) = v50 + 1;
        }
      }
LABEL_8:
      if ( v10 == (_DWORD)++v11 )
        goto LABEL_19;
    }
    if ( (_BYTE)v12 )
      goto LABEL_8;
    if ( (*(_BYTE *)(v14 + 3) & 0x40) != 0 || v45 <= 0x3FFFFFFE && (*(_WORD *)(v14 + 2) & 0xFF0) != 0 )
      goto LABEL_43;
    ++v11;
    *(_BYTE *)(v14 + 3) |= 0x40u;
    v12 = 1;
  }
  while ( v10 != (_DWORD)v11 );
LABEL_19:
  v16 = (unsigned int)v50;
  v6 = a2;
  v17 = v12;
  v8 = v4 & (v12 ^ 1);
  if ( (_DWORD)v50 )
  {
    v46 = v4 & (v12 ^ 1);
    v18 = v7;
    while ( 1 )
    {
      v21 = *(unsigned int *)&v49[4 * v16 - 4];
      v19 = 40 * v21 + *(_QWORD *)(v18 + 32);
      if ( (*(_BYTE *)(v19 + 3) & 0x20) == 0 )
        goto LABEL_23;
      if ( (unsigned int)*(unsigned __int16 *)(v18 + 68) - 1 > 1 || (int)sub_2E890A0(v18, v21, 0) < 0 )
      {
        sub_2E8A650(v18, v21);
        v20 = (_DWORD)v50 == 1;
        v16 = (unsigned int)(v50 - 1);
        LODWORD(v50) = v50 - 1;
        if ( v20 )
        {
LABEL_27:
          v7 = v18;
          v8 = v46;
          break;
        }
      }
      else
      {
        v19 = 40 * v21 + *(_QWORD *)(v18 + 32);
LABEL_23:
        *(_BYTE *)(v19 + 3) &= ~0x40u;
        v20 = (_DWORD)v50 == 1;
        v16 = (unsigned int)(v50 - 1);
        LODWORD(v50) = v50 - 1;
        if ( v20 )
          goto LABEL_27;
      }
    }
  }
LABEL_28:
  if ( (_BYTE)v8 )
  {
    v52.m128i_i32[2] = v6;
    v17 = v8;
    v52.m128i_i64[0] = 1610612736;
    v53 = 0;
    v54 = 0;
    v55 = 0;
    sub_2E8F270(v7, &v52);
  }
LABEL_30:
  if ( v49 != v51 )
    _libc_free((unsigned __int64)v49);
  return v17;
}
