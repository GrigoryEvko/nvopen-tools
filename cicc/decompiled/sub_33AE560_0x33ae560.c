// Function: sub_33AE560
// Address: 0x33ae560
//
void __fastcall sub_33AE560(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 *v5; // r13
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rsi
  __int128 v11; // rax
  __int64 v12; // rsi
  __int128 v13; // rax
  __int64 v14; // r9
  __int64 *v15; // rdx
  unsigned int v16; // eax
  __int64 v17; // r13
  unsigned int v18; // ecx
  __int64 v19; // r8
  int v20; // r15d
  __int64 v21; // rax
  _BYTE *v22; // rdx
  int v23; // ebx
  _BYTE *v24; // rdx
  __int64 v25; // r15
  int v26; // edx
  _QWORD *v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rdi
  __int64 (__fastcall *v31)(__int64, __int64, unsigned int); // rax
  _DWORD *v32; // rax
  int v33; // r10d
  int v34; // edx
  unsigned __int16 v35; // ax
  __int128 v36; // rax
  int v37; // r9d
  __int64 v38; // r13
  int v39; // edx
  int v40; // r15d
  _QWORD *v41; // rax
  __int64 (__fastcall *v42)(__int64, __int64, __int64, __int64); // [rsp+0h] [rbp-E0h]
  __int64 v43; // [rsp+8h] [rbp-D8h]
  int v44; // [rsp+8h] [rbp-D8h]
  __int128 v45; // [rsp+10h] [rbp-D0h]
  __int128 v46; // [rsp+20h] [rbp-C0h]
  int v47; // [rsp+20h] [rbp-C0h]
  __int64 v48; // [rsp+58h] [rbp-88h] BYREF
  int v49; // [rsp+60h] [rbp-80h] BYREF
  __int64 v50; // [rsp+68h] [rbp-78h]
  __int64 v51; // [rsp+70h] [rbp-70h] BYREF
  int v52; // [rsp+78h] [rbp-68h]
  _BYTE *v53; // [rsp+80h] [rbp-60h] BYREF
  __int64 v54; // [rsp+88h] [rbp-58h]
  _BYTE v55[80]; // [rsp+90h] [rbp-50h] BYREF

  v3 = a1;
  v4 = *(_QWORD *)(a1 + 864);
  v5 = *(__int64 **)(a2 + 8);
  v6 = *(_QWORD *)(v4 + 16);
  v7 = sub_2E79000(*(__int64 **)(v4 + 40));
  v51 = 0;
  v49 = sub_2D5BAE0(v6, v7, v5, 0);
  v8 = *(_QWORD *)a1;
  v50 = v9;
  v52 = *(_DWORD *)(a1 + 848);
  if ( v8 )
  {
    if ( &v51 != (__int64 *)(v8 + 48) )
    {
      v10 = *(_QWORD *)(v8 + 48);
      v51 = v10;
      if ( v10 )
        sub_B96E90((__int64)&v51, v10, 1);
    }
  }
  *(_QWORD *)&v11 = sub_338B750(a1, *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v46 = v11;
  v12 = *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  *(_QWORD *)&v13 = sub_338B750(a1, v12);
  v45 = v13;
  *(_QWORD *)&v13 = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v15 = *(__int64 **)(v13 + 24);
  v16 = *(_DWORD *)(v13 + 32);
  if ( v16 > 0x40 )
  {
    v17 = *v15;
  }
  else
  {
    v17 = 0;
    if ( v16 )
      v17 = (__int64)((_QWORD)v15 << (64 - (unsigned __int8)v16)) >> (64 - (unsigned __int8)v16);
  }
  if ( (_WORD)v49 )
  {
    if ( (unsigned __int16)(v49 - 176) > 0x34u )
    {
      v18 = word_4456340[(unsigned __int16)v49 - 1];
      goto LABEL_11;
    }
LABEL_23:
    v43 = *(_QWORD *)(a1 + 864);
    v42 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v6 + 72LL);
    v29 = sub_2E79000(*(__int64 **)(v43 + 40));
    v30 = v29;
    if ( v42 == sub_2FE4D20 )
    {
      v31 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v6 + 32LL);
      if ( v31 == sub_2D42F30 )
      {
        v32 = sub_AE2980(v30, 0);
        v33 = v43;
        v34 = v32[1];
        v35 = 2;
        if ( v34 != 1 )
        {
          v35 = 3;
          if ( v34 != 2 )
          {
            v35 = 4;
            if ( v34 != 4 )
            {
              v35 = 5;
              if ( v34 != 8 )
              {
                v35 = 6;
                if ( v34 != 16 )
                {
                  v35 = 7;
                  if ( v34 != 32 )
                  {
                    v35 = 8;
                    if ( v34 != 64 )
                      v35 = 9 * (v34 == 128);
                  }
                }
              }
            }
          }
        }
      }
      else
      {
        v35 = v31(v6, v30, 0);
        v33 = v43;
      }
    }
    else
    {
      v35 = ((__int64 (__fastcall *)(__int64, __int64))v42)(v6, v29);
      v33 = v43;
    }
    v44 = v33;
    *(_QWORD *)&v36 = sub_3401400(v33, v17, (unsigned int)&v51, v35, 0, 0, 0);
    v38 = sub_340F900(v44, 166, (unsigned int)&v51, v49, v50, v37, v46, v45, v36);
    v40 = v39;
    v53 = (_BYTE *)a2;
    v41 = sub_337DC20(v3 + 8, (__int64 *)&v53);
    *v41 = v38;
    v28 = v51;
    *((_DWORD *)v41 + 2) = v40;
    if ( v28 )
      goto LABEL_21;
    return;
  }
  if ( sub_3007100((__int64)&v49) )
    goto LABEL_23;
  v18 = sub_3007130((__int64)&v49, v12);
LABEL_11:
  v53 = v55;
  v54 = 0x800000000LL;
  if ( v18 )
  {
    v19 = v18 + (unsigned int)((v18 + v17) % v18) - 1;
    v20 = (v18 + v17) % v18;
    v21 = 0;
    v22 = v55;
    v23 = v18 + v20 - 1;
    while ( 1 )
    {
      *(_DWORD *)&v22[4 * v21] = v20;
      v21 = (unsigned int)(v54 + 1);
      LODWORD(v54) = v54 + 1;
      if ( v23 == v20 )
        break;
      if ( v21 + 1 > (unsigned __int64)HIDWORD(v54) )
      {
        sub_C8D5F0((__int64)&v53, v55, v21 + 1, 4u, v19, v14);
        v21 = (unsigned int)v54;
      }
      v22 = v53;
      ++v20;
    }
    v3 = a1;
    v24 = v53;
  }
  else
  {
    v21 = 0;
    v24 = v55;
  }
  v25 = sub_33FCE10(
          *(_QWORD *)(v3 + 864),
          v49,
          v50,
          (unsigned int)&v51,
          v46,
          DWORD2(v46),
          v45,
          *((__int64 *)&v45 + 1),
          (__int64)v24,
          v21);
  v47 = v26;
  v48 = a2;
  v27 = sub_337DC20(v3 + 8, &v48);
  *v27 = v25;
  *((_DWORD *)v27 + 2) = v47;
  if ( v53 != v55 )
    _libc_free((unsigned __int64)v53);
  v28 = v51;
  if ( v51 )
LABEL_21:
    sub_B91220((__int64)&v51, v28);
}
