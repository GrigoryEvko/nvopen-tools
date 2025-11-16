// Function: sub_1E1B440
// Address: 0x1e1b440
//
__int64 __fastcall sub_1E1B440(__int64 a1, int a2, _QWORD *a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  _QWORD *v6; // r10
  __int64 v9; // r8
  unsigned int v10; // eax
  __int16 v11; // di
  _WORD *v12; // rax
  __int16 *v13; // rdx
  unsigned __int16 v14; // di
  __int16 *v15; // r14
  unsigned __int16 *v16; // rax
  unsigned int v17; // ebx
  unsigned __int16 *v18; // rsi
  unsigned __int16 *v19; // rax
  int v20; // edx
  char v21; // r11
  __int64 v22; // r15
  __int64 v23; // rbx
  unsigned int v24; // r14d
  __int64 v25; // r8
  __int64 v26; // rdx
  int v27; // eax
  __int64 v28; // rax
  bool v29; // zf
  __int64 v30; // rsi
  __int64 v31; // rdx
  char v32; // al
  unsigned __int16 *v34; // rdi
  int v35; // edx
  unsigned __int16 *v36; // rdi
  int v37; // esi
  unsigned __int16 *v38; // rdx
  int v39; // ecx
  unsigned __int16 *v40; // rsi
  unsigned __int16 *v41; // rdx
  unsigned __int16 *v42; // rdx
  unsigned __int16 v43; // ax
  unsigned __int16 *v44; // rdx
  __int64 v45; // rax
  __int16 v46; // ax
  _QWORD *v47; // [rsp+0h] [rbp-B0h]
  __int64 v48; // [rsp+8h] [rbp-A8h]
  char v49; // [rsp+8h] [rbp-A8h]
  unsigned __int8 v50; // [rsp+20h] [rbp-90h]
  char v51; // [rsp+24h] [rbp-8Ch]
  _BYTE *v52; // [rsp+30h] [rbp-80h] BYREF
  __int64 v53; // [rsp+38h] [rbp-78h]
  _BYTE v54[16]; // [rsp+40h] [rbp-70h] BYREF
  __m128i v55; // [rsp+50h] [rbp-60h] BYREF
  __int64 v56; // [rsp+60h] [rbp-50h]
  __int64 v57; // [rsp+68h] [rbp-48h]
  __int64 v58; // [rsp+70h] [rbp-40h]

  v6 = a3;
  v51 = a4;
  if ( a2 > 0 )
  {
    if ( !a3 )
      BUG();
    a6 = a3[1];
    v9 = a3[7];
    v10 = *(_DWORD *)(a6 + 24LL * (unsigned int)a2 + 16);
    v11 = a2 * (v10 & 0xF);
    v12 = (_WORD *)(v9 + 2LL * (v10 >> 4));
    v13 = v12 + 1;
    v14 = *v12 + v11;
LABEL_4:
    v15 = v13;
    while ( v15 )
    {
      v16 = (unsigned __int16 *)(v6[6] + 4LL * v14);
      a4 = *v16;
      v17 = v16[1];
      if ( (_WORD)a4 )
      {
        while ( 1 )
        {
          v18 = (unsigned __int16 *)(v9 + 2LL * *(unsigned int *)(a6 + 24LL * (unsigned __int16)a4 + 8));
LABEL_8:
          v19 = v18;
          if ( v18 )
            break;
LABEL_12:
          a4 = v17;
          if ( !(_WORD)v17 )
            goto LABEL_60;
          v17 = 0;
        }
        while ( a2 == (unsigned __int16)a4 )
        {
          v20 = *v19;
          v18 = 0;
          ++v19;
          if ( !(_WORD)v20 )
            goto LABEL_8;
          a4 = (unsigned int)(v20 + a4);
          if ( !v19 )
            goto LABEL_12;
        }
        v21 = 1;
        goto LABEL_15;
      }
LABEL_60:
      v46 = *v15;
      v13 = 0;
      ++v15;
      v14 += v46;
      if ( !v46 )
        goto LABEL_4;
    }
  }
  v21 = 0;
LABEL_15:
  v22 = *(unsigned int *)(a1 + 40);
  v52 = v54;
  v53 = 0x400000000LL;
  if ( (_DWORD)v22 )
  {
    v23 = 0;
    v24 = 0;
    while ( 1 )
    {
LABEL_17:
      while ( 1 )
      {
        v25 = (unsigned int)v23;
        v26 = *(_QWORD *)(a1 + 32) + 40 * v23;
        if ( !*(_BYTE *)v26 )
        {
          a4 = *(unsigned __int8 *)(v26 + 3);
          if ( (a4 & 0x10) != 0 )
          {
            v27 = *(_DWORD *)(v26 + 8);
            if ( v27 )
              break;
          }
        }
LABEL_24:
        if ( v22 == ++v23 )
          goto LABEL_25;
      }
      if ( a2 != v27 )
      {
        if ( v21 )
        {
          a4 = (a4 & 0x10) != 0;
          v50 = a4 & (*(_BYTE *)(v26 + 3) >> 6);
          if ( v50 )
          {
            if ( v27 > 0 )
            {
              v48 = v6[7];
              v34 = (unsigned __int16 *)(v48 + 2LL * *(unsigned int *)(v6[1] + 24LL * (unsigned int)a2 + 8));
              v35 = *v34;
              v36 = v34 + 1;
              v37 = v35 + a2;
              if ( !(_WORD)v35 )
                v36 = 0;
LABEL_40:
              v38 = v36;
              if ( v36 )
              {
                while ( v27 != (unsigned __int16)v37 )
                {
                  v39 = *v38;
                  v36 = 0;
                  ++v38;
                  a6 = (unsigned int)(v39 + v37);
                  if ( !(_WORD)v39 )
                    goto LABEL_40;
                  v37 += v39;
                  if ( !v38 )
                    goto LABEL_44;
                }
                v24 = v50;
                goto LABEL_33;
              }
LABEL_44:
              v40 = 0;
              v41 = (unsigned __int16 *)(v48 + 2LL * *(unsigned int *)(v6[1] + 24LL * (unsigned int)v27 + 8));
              a4 = *v41;
              v42 = v41 + 1;
              v43 = a4 + v27;
              if ( (_WORD)a4 )
                v40 = v42;
LABEL_49:
              v44 = v40;
              while ( v44 )
              {
                a4 = v43;
                if ( a2 == v43 )
                {
                  v45 = (unsigned int)v53;
                  if ( (unsigned int)v53 >= HIDWORD(v53) )
                  {
                    v47 = v6;
                    v49 = v21;
                    sub_16CD150((__int64)&v52, v54, 0, 4, v23, a6);
                    v45 = (unsigned int)v53;
                    v6 = v47;
                    v21 = v49;
                    v25 = (unsigned int)v23;
                  }
                  ++v23;
                  *(_DWORD *)&v52[4 * v45] = v25;
                  LODWORD(v53) = v53 + 1;
                  if ( v22 != v23 )
                    goto LABEL_17;
                  goto LABEL_25;
                }
                a4 = *v44;
                v40 = 0;
                ++v44;
                v43 += a4;
                if ( !(_WORD)a4 )
                  goto LABEL_49;
              }
            }
          }
        }
        goto LABEL_24;
      }
      a4 = (unsigned int)a4 | 0x40;
      ++v23;
      v24 = 1;
      *(_BYTE *)(v26 + 3) = a4;
      if ( v22 == v23 )
      {
LABEL_25:
        v28 = (unsigned int)v53;
        if ( (_DWORD)v53 )
        {
          do
          {
            while ( 1 )
            {
              v30 = *(unsigned int *)&v52[4 * v28 - 4];
              v31 = *(_QWORD *)(a1 + 32) + 40 * v30;
              v32 = *(_BYTE *)(v31 + 3);
              if ( (v32 & 0x20) != 0 )
                break;
              *(_BYTE *)(v31 + 3) = v32 & 0xBF;
              v29 = (_DWORD)v53 == 1;
              v28 = (unsigned int)(v53 - 1);
              LODWORD(v53) = v53 - 1;
              if ( v29 )
                goto LABEL_30;
            }
            sub_1E16C90(a1, v30, v31, a4, v25, (_BYTE *)a6);
            v29 = (_DWORD)v53 == 1;
            v28 = (unsigned int)(v53 - 1);
            LODWORD(v53) = v53 - 1;
          }
          while ( !v29 );
        }
LABEL_30:
        if ( v51 != 1 || (_BYTE)v24 )
          goto LABEL_33;
        goto LABEL_32;
      }
    }
  }
  if ( v51 )
  {
LABEL_32:
    v55.m128i_i32[2] = a2;
    v24 = 1;
    v55.m128i_i64[0] = 1879048192;
    v56 = 0;
    v57 = 0;
    v58 = 0;
    sub_1E1AFD0(a1, &v55);
LABEL_33:
    if ( v52 != v54 )
      _libc_free((unsigned __int64)v52);
  }
  else
  {
    return 0;
  }
  return v24;
}
