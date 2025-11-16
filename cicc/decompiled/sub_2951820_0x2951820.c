// Function: sub_2951820
// Address: 0x2951820
//
__int64 __fastcall sub_2951820(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 v4; // rdx
  __int64 v5; // rdx
  __int64 v6; // rbx
  unsigned int v7; // r15d
  __int64 v8; // r14
  unsigned __int64 v9; // r12
  unsigned __int64 v10; // r10
  __int64 v11; // rax
  _QWORD *v12; // r14
  unsigned __int64 v13; // r10
  unsigned __int8 *v14; // rsi
  __int64 v15; // rax
  char v16; // al
  char v17; // al
  int v18; // eax
  __int64 v19; // rdx
  bool v20; // al
  __int64 v21; // r14
  unsigned __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rdx
  unsigned __int64 v25; // r10
  unsigned __int64 v26; // rax
  __int64 v27; // rax
  bool v28; // zf
  char v29; // al
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  char v39; // al
  unsigned __int64 v40; // [rsp+0h] [rbp-1A0h]
  __int64 v42; // [rsp+18h] [rbp-188h]
  unsigned __int64 v43; // [rsp+20h] [rbp-180h]
  unsigned __int64 v44; // [rsp+20h] [rbp-180h]
  unsigned __int64 v45; // [rsp+20h] [rbp-180h]
  char v46; // [rsp+20h] [rbp-180h]
  unsigned __int64 v47; // [rsp+20h] [rbp-180h]
  unsigned __int64 v48; // [rsp+20h] [rbp-180h]
  unsigned __int64 v49; // [rsp+20h] [rbp-180h]
  unsigned __int64 v50; // [rsp+20h] [rbp-180h]
  unsigned __int64 v51; // [rsp+20h] [rbp-180h]
  unsigned __int64 v52; // [rsp+38h] [rbp-168h]
  __int64 v53; // [rsp+38h] [rbp-168h]
  __int64 v54; // [rsp+38h] [rbp-168h]
  int v55; // [rsp+4Ch] [rbp-154h]
  unsigned __int8 **v56; // [rsp+50h] [rbp-150h]
  __int64 v57; // [rsp+58h] [rbp-148h]
  unsigned __int64 v58; // [rsp+60h] [rbp-140h] BYREF
  unsigned int v59; // [rsp+68h] [rbp-138h]
  unsigned __int64 v60; // [rsp+70h] [rbp-130h] BYREF
  __int64 v61; // [rsp+78h] [rbp-128h]
  _BYTE v62[64]; // [rsp+80h] [rbp-120h] BYREF
  _BYTE *v63; // [rsp+C0h] [rbp-E0h]
  __int64 v64; // [rsp+C8h] [rbp-D8h]
  _BYTE v65[128]; // [rsp+D0h] [rbp-D0h] BYREF
  __int64 v66; // [rsp+150h] [rbp-50h]
  __int16 v67; // [rsp+158h] [rbp-48h]
  __int64 v68; // [rsp+160h] [rbp-40h]

  *a3 = 0;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v4 = *(_QWORD *)(a2 - 8);
  else
    v4 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v56 = (unsigned __int8 **)(v4 + 32);
  v6 = sub_BB5290(a2) & 0xFFFFFFFFFFFFFFF9LL | 4;
  v55 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( v55 != 1 )
  {
    v7 = 1;
    v42 = 0;
    while ( 1 )
    {
      v8 = v7++;
      v9 = v6 & 0xFFFFFFFFFFFFFFF8LL;
      v10 = v6 & 0xFFFFFFFFFFFFFFF8LL;
      v57 = (v6 >> 1) & 3;
      if ( ((v6 >> 1) & 3) == 0 )
      {
        if ( *(_BYTE *)(a1 + 48) )
        {
          v11 = *(_QWORD *)(a2 + 32 * (v8 - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
          v12 = *(_QWORD **)(v11 + 24);
          if ( *(_DWORD *)(v11 + 32) > 0x40u )
            v12 = (_QWORD *)*v12;
          if ( v12 )
          {
            *a3 = 1;
            v31 = 16LL * (unsigned int)v12 + sub_AE4AC0(*(_QWORD *)a1, v6 & 0xFFFFFFFFFFFFFFF8LL) + 24;
            v32 = *(_QWORD *)v31;
            LOBYTE(v31) = *(_BYTE *)(v31 + 8);
            v60 = v32;
            LOBYTE(v61) = v31;
            v33 = sub_CA1930(&v60);
            v10 = v6 & 0xFFFFFFFFFFFFFFF8LL;
            v42 += v33;
          }
        }
        goto LABEL_13;
      }
      if ( !v6 )
        break;
      if ( ((v6 >> 1) & 3) == 2 )
      {
        if ( v9 )
        {
          v17 = sub_BCEA30(v6 & 0xFFFFFFFFFFFFFFF8LL);
          v10 = v6 & 0xFFFFFFFFFFFFFFF8LL;
          if ( v17 )
            goto LABEL_16;
          goto LABEL_29;
        }
        v52 = 0;
        goto LABEL_53;
      }
      v52 = v6 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (_DWORD)v57 == 1 )
      {
        if ( v9 )
        {
          v29 = sub_BCEA30(*(_QWORD *)(v9 + 24));
          v10 = v6 & 0xFFFFFFFFFFFFFFF8LL;
          if ( v29 )
            goto LABEL_24;
          goto LABEL_29;
        }
LABEL_53:
        v14 = *v56;
LABEL_20:
        v15 = sub_BCBAE0(0, v14, v5);
        v16 = sub_BCEA30(v15);
        v10 = v52;
        if ( v16 )
          goto LABEL_21;
        goto LABEL_29;
      }
      v38 = sub_BCBAE0(v6 & 0xFFFFFFFFFFFFFFF8LL, *v56, v5);
      v39 = sub_BCEA30(v38);
      v10 = v6 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v39 )
        goto LABEL_14;
LABEL_29:
      v18 = *(_DWORD *)(a2 + 4);
      v43 = v10;
      v60 = (unsigned __int64)v62;
      v19 = *(_QWORD *)(a2 + 32 * (v8 - (v18 & 0x7FFFFFF)));
      v61 = 0x800000000LL;
      v64 = 0x1000000000LL;
      v66 = a2 + 24;
      v53 = v19;
      v63 = v65;
      v67 = 0;
      v68 = sub_B43CC0(a2);
      v20 = sub_B4DE30(a2);
      sub_2950CC0((__int64)&v58, (__int64)&v60, v53, 0, 0, v20);
      v10 = v43;
      if ( v59 > 0x40 )
      {
        v54 = *(_QWORD *)v58;
        j_j___libc_free_0_0(v58);
        v10 = v43;
      }
      else
      {
        v54 = 0;
        if ( v59 )
          v54 = (__int64)(v58 << (64 - (unsigned __int8)v59)) >> (64 - (unsigned __int8)v59);
      }
      if ( v63 != v65 )
      {
        v44 = v10;
        _libc_free((unsigned __int64)v63);
        v10 = v44;
      }
      if ( (_BYTE *)v60 != v62 )
      {
        v45 = v10;
        _libc_free(v60);
        v10 = v45;
      }
      if ( v54 )
      {
        *a3 = 1;
        v21 = *(_QWORD *)a1;
        if ( v6 )
        {
          if ( v57 == 2 )
          {
            v22 = v6 & 0xFFFFFFFFFFFFFFF8LL;
            if ( v9 )
              goto LABEL_40;
LABEL_61:
            v50 = v10;
            v36 = sub_BCBAE0(v6 & 0xFFFFFFFFFFFFFFF8LL, *v56, v5);
            v10 = v50;
            v22 = v36;
LABEL_40:
            v40 = v10;
            v46 = sub_AE5020(v21, v22);
            v23 = sub_9208B0(v21, v22);
            v25 = v40;
            v61 = v24;
            v26 = (((unsigned __int64)(v23 + 7) >> 3) + (1LL << v46) - 1) >> v46 << v46;
LABEL_41:
            v47 = v25;
            LOBYTE(v61) = v24;
            v60 = v54 * v26;
            v27 = sub_CA1930(&v60);
            v10 = v47;
            v42 += v27;
            goto LABEL_13;
          }
          if ( (_DWORD)v57 != 1 )
            goto LABEL_61;
          if ( v9 )
          {
            v22 = *(_QWORD *)(v9 + 24);
          }
          else
          {
            v51 = v10;
            v37 = sub_BCBAE0(0, *v56, v5);
            v10 = v51;
            v22 = v37;
          }
        }
        else
        {
          v49 = v10;
          v35 = sub_BCBAE0(v9, *v56, v5);
          v10 = v49;
          v22 = v35;
          if ( v57 != 1 )
            goto LABEL_40;
        }
        v48 = v10;
        v34 = sub_9208B0(v21, v22);
        v25 = v48;
        v61 = v24;
        v26 = (unsigned __int64)(v34 + 7) >> 3;
        goto LABEL_41;
      }
LABEL_13:
      if ( !v6 )
        goto LABEL_21;
LABEL_14:
      if ( v57 != 2 )
      {
        if ( v57 == 1 && v9 )
        {
LABEL_24:
          v10 = *(_QWORD *)(v9 + 24);
          goto LABEL_16;
        }
LABEL_21:
        v10 = sub_BCBAE0(v6 & 0xFFFFFFFFFFFFFFF8LL, *v56, v5);
        goto LABEL_16;
      }
      if ( !v9 )
        goto LABEL_21;
LABEL_16:
      v5 = *(unsigned __int8 *)(v10 + 8);
      if ( (_BYTE)v5 == 16 )
      {
        v6 = *(_QWORD *)(v10 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
      }
      else
      {
        v13 = v10 & 0xFFFFFFFFFFFFFFF9LL;
        if ( (unsigned int)(unsigned __int8)v5 - 17 > 1 )
        {
          v28 = (_BYTE)v5 == 15;
          v5 = 0;
          if ( v28 )
            v5 = v13;
          v6 = v5;
        }
        else
        {
          v6 = v13 | 2;
        }
      }
      v56 += 4;
      if ( v55 == v7 )
        return v42;
    }
    v52 = 0;
    v14 = *v56;
    goto LABEL_20;
  }
  return 0;
}
