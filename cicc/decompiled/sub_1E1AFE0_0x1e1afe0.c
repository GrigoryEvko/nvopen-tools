// Function: sub_1E1AFE0
// Address: 0x1e1afe0
//
__int64 __fastcall sub_1E1AFE0(__int64 a1, int a2, _QWORD *a3, int a4, __int64 a5, unsigned __int64 a6)
{
  int v9; // ebx
  __int64 v10; // r10
  _WORD *v11; // rax
  unsigned __int16 *v12; // rdx
  unsigned __int16 *v13; // r13
  unsigned __int16 *v14; // rax
  unsigned __int16 v15; // cx
  unsigned __int16 v16; // di
  __int16 *v17; // rsi
  __int16 *v18; // rax
  __int16 v19; // dx
  char v20; // r10
  __int64 v21; // r13
  __int64 v22; // rbx
  int v23; // r11d
  int v24; // r15d
  __int64 v25; // rcx
  unsigned __int8 v26; // al
  int v27; // edx
  _BOOL4 v28; // eax
  __int64 v29; // rax
  unsigned int v30; // r15d
  bool v31; // zf
  __int64 v32; // rsi
  __int64 v33; // rdx
  char v34; // al
  char v36; // al
  unsigned __int16 *v37; // rcx
  int v38; // esi
  unsigned __int16 *v39; // rcx
  int v40; // edi
  unsigned __int16 *v41; // rcx
  int v42; // esi
  unsigned __int16 *v43; // rsi
  unsigned __int16 *v44; // rax
  unsigned __int16 *v45; // rax
  unsigned __int16 v46; // dx
  unsigned __int16 *v47; // rax
  unsigned __int16 v48; // di
  __int64 v49; // rax
  int v50; // eax
  unsigned __int8 v51; // [rsp+8h] [rbp-B8h]
  __int64 v52; // [rsp+10h] [rbp-B0h]
  char v53; // [rsp+10h] [rbp-B0h]
  _QWORD *v54; // [rsp+20h] [rbp-A0h]
  char v55; // [rsp+34h] [rbp-8Ch]
  _BYTE *v56; // [rsp+40h] [rbp-80h] BYREF
  __int64 v57; // [rsp+48h] [rbp-78h]
  _BYTE v58[16]; // [rsp+50h] [rbp-70h] BYREF
  __m128i v59; // [rsp+60h] [rbp-60h] BYREF
  __int64 v60; // [rsp+70h] [rbp-50h]
  __int64 v61; // [rsp+78h] [rbp-48h]
  __int64 v62; // [rsp+80h] [rbp-40h]

  v9 = a4;
  v55 = a4;
  if ( a2 > 0 )
  {
    if ( !a3 )
      BUG();
    v10 = a3[1];
    a6 = a3[7];
    a5 = a2 * (*(_DWORD *)(v10 + 24LL * (unsigned int)a2 + 16) & 0xFu);
    v11 = (_WORD *)(a6 + 2LL * (*(_DWORD *)(v10 + 24LL * (unsigned int)a2 + 16) >> 4));
    v12 = v11 + 1;
    LOWORD(a5) = *v11 + a2 * (*(_WORD *)(v10 + 24LL * (unsigned int)a2 + 16) & 0xF);
LABEL_4:
    v13 = v12;
    while ( v13 )
    {
      v14 = (unsigned __int16 *)(a3[6] + 4LL * (unsigned __int16)a5);
      v15 = *v14;
      v16 = v14[1];
      if ( *v14 )
      {
        while ( 1 )
        {
          v17 = (__int16 *)(a6 + 2LL * *(unsigned int *)(v10 + 24LL * v15 + 8));
LABEL_8:
          v18 = v17;
          if ( v17 )
            break;
LABEL_12:
          v15 = v16;
          if ( !v16 )
            goto LABEL_63;
          v16 = 0;
        }
        while ( a2 == v15 )
        {
          v19 = *v18;
          v17 = 0;
          ++v18;
          if ( !v19 )
            goto LABEL_8;
          v15 += v19;
          if ( !v18 )
            goto LABEL_12;
        }
        v20 = 1;
        goto LABEL_15;
      }
LABEL_63:
      v50 = *v13;
      v12 = 0;
      ++v13;
      a5 = (unsigned int)(v50 + a5);
      if ( !(_WORD)v50 )
        goto LABEL_4;
    }
  }
  v20 = 0;
LABEL_15:
  v21 = *(unsigned int *)(a1 + 40);
  v56 = v58;
  v57 = 0x400000000LL;
  if ( (_DWORD)v21 )
  {
    v54 = a3;
    v22 = 0;
    v23 = 0;
    while ( 1 )
    {
LABEL_17:
      v24 = v22;
      v25 = *(_QWORD *)(a1 + 32) + 40 * v22;
      if ( *(_BYTE *)v25 )
        goto LABEL_26;
      v26 = *(_BYTE *)(v25 + 3);
      if ( (v26 & 0x10) != 0 )
        goto LABEL_26;
      if ( (*(_BYTE *)(v25 + 4) & 1) != 0 )
        goto LABEL_26;
      if ( (*(_BYTE *)(v25 + 4) & 8) != 0 )
        goto LABEL_26;
      v27 = *(_DWORD *)(v25 + 8);
      if ( !v27 )
        goto LABEL_26;
      if ( a2 == v27 )
      {
        if ( (_BYTE)v23 )
          goto LABEL_26;
        v36 = v26 >> 6;
        if ( (v36 & 1) != 0 )
        {
          v30 = v36 & 1;
          goto LABEL_34;
        }
        if ( a2 > 0 && (*(_WORD *)(v25 + 2) & 0xFF0) != 0 )
        {
          v30 = 1;
          goto LABEL_34;
        }
        ++v22;
        *(_BYTE *)(v25 + 3) |= 0x40u;
        v23 = 1;
        if ( v21 == v22 )
        {
LABEL_27:
          v29 = (unsigned int)v57;
          v30 = v23;
          v9 = v23 ^ 1;
          LOBYTE(v9) = v55 & (v23 ^ 1);
          if ( (_DWORD)v57 )
          {
            do
            {
              while ( 1 )
              {
                v32 = *(unsigned int *)&v56[4 * v29 - 4];
                v33 = *(_QWORD *)(a1 + 32) + 40 * v32;
                v34 = *(_BYTE *)(v33 + 3);
                if ( (v34 & 0x20) != 0 )
                  break;
                *(_BYTE *)(v33 + 3) = v34 & 0xBF;
                v31 = (_DWORD)v57 == 1;
                v29 = (unsigned int)(v57 - 1);
                LODWORD(v57) = v57 - 1;
                if ( v31 )
                  goto LABEL_32;
              }
              sub_1E16C90(a1, v32, v33, v25, a5, (_BYTE *)a6);
              v31 = (_DWORD)v57 == 1;
              v29 = (unsigned int)(v57 - 1);
              LODWORD(v57) = v57 - 1;
            }
            while ( !v31 );
          }
          goto LABEL_32;
        }
      }
      else
      {
        if ( v20 )
        {
          v28 = (v26 & 0x40) != 0;
          if ( v28 && v27 > 0 )
          {
            a5 = 0;
            v52 = v54[7];
            v37 = (unsigned __int16 *)(v52 + 2LL * *(unsigned int *)(v54[1] + 24LL * (unsigned int)a2 + 8));
            v38 = *v37;
            v39 = v37 + 1;
            v40 = v38 + a2;
            if ( (_WORD)v38 )
              a5 = (__int64)v39;
LABEL_44:
            v41 = (unsigned __int16 *)a5;
            if ( a5 )
            {
              while ( v27 != (unsigned __int16)v40 )
              {
                v42 = *v41;
                a5 = 0;
                ++v41;
                a6 = (unsigned int)(v42 + v40);
                if ( !(_WORD)v42 )
                  goto LABEL_44;
                v40 += v42;
                if ( !v41 )
                  goto LABEL_48;
              }
              v30 = v28;
              goto LABEL_34;
            }
LABEL_48:
            v43 = 0;
            v44 = (unsigned __int16 *)(v52 + 2LL * *(unsigned int *)(v54[1] + 24LL * (unsigned int)v27 + 8));
            v25 = *v44;
            v45 = v44 + 1;
            v46 = v25 + v27;
            if ( (_WORD)v25 )
              v43 = v45;
            while ( 1 )
            {
              v47 = v43;
              if ( !v43 )
                break;
              v25 = v46;
              if ( a2 == v46 )
              {
LABEL_55:
                v49 = (unsigned int)v57;
                if ( (unsigned int)v57 >= HIDWORD(v57) )
                {
                  v51 = v23;
                  v53 = v20;
                  sub_16CD150((__int64)&v56, v58, 0, 4, a5, a6);
                  v49 = (unsigned int)v57;
                  v23 = v51;
                  v20 = v53;
                }
                ++v22;
                *(_DWORD *)&v56[4 * v49] = v24;
                LODWORD(v57) = v57 + 1;
                if ( v21 != v22 )
                  goto LABEL_17;
                goto LABEL_27;
              }
              while ( 1 )
              {
                v25 = *v47;
                v43 = 0;
                ++v47;
                v48 = v25 + v46;
                if ( !(_WORD)v25 )
                  break;
                v46 += v25;
                if ( !v47 )
                  goto LABEL_26;
                v25 = v48;
                if ( a2 == v48 )
                  goto LABEL_55;
              }
            }
          }
        }
LABEL_26:
        if ( v21 == ++v22 )
          goto LABEL_27;
      }
    }
  }
  v30 = 0;
LABEL_32:
  if ( (_BYTE)v9 )
  {
    v59.m128i_i32[2] = a2;
    v30 = v9;
    v59.m128i_i64[0] = 1610612736;
    v60 = 0;
    v61 = 0;
    v62 = 0;
    sub_1E1AFD0(a1, &v59);
  }
LABEL_34:
  if ( v56 != v58 )
    _libc_free((unsigned __int64)v56);
  return v30;
}
