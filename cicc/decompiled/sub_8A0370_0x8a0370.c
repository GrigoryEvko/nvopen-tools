// Function: sub_8A0370
// Address: 0x8a0370
//
__int64 *__fastcall sub_8A0370(__int64 a1, __m128i **a2, int a3, __int64 *a4, __int64 a5, int a6, unsigned int a7)
{
  __int64 v7; // r15
  __int64 **v8; // r12
  char v10; // dl
  __int64 v11; // rax
  __int64 v12; // r14
  int v13; // r13d
  __m128i *v14; // rsi
  char v15; // al
  __int16 v16; // r11
  __int64 v17; // r9
  __int16 v18; // r11
  __m128i *v19; // r8
  __int64 v20; // rcx
  __m128i *v21; // rax
  __int64 *v22; // rbx
  unsigned __int8 *v23; // rdi
  __int64 *v24; // rax
  __m128i *v25; // rdi
  __int64 v27; // rdx
  __int64 v28; // r12
  __int64 v29; // rbx
  __int16 v30; // r13
  char v31; // al
  __int64 v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rbx
  __int64 *v35; // rax
  int v36; // eax
  __int64 v37; // rax
  unsigned __int8 v38; // di
  char v39; // r13
  _QWORD *v40; // rax
  bool v41; // zf
  __int64 v42; // r13
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // rdx
  __int64 v47; // rax
  int v48; // eax
  __int64 v49; // rdx
  __int64 v50; // rdi
  _BOOL4 v51; // eax
  __int64 **v52; // [rsp+0h] [rbp-90h]
  int v53; // [rsp+Ch] [rbp-84h]
  __int16 v54; // [rsp+Ch] [rbp-84h]
  __int16 v57; // [rsp+1Ch] [rbp-74h]
  bool v58; // [rsp+1Ch] [rbp-74h]
  unsigned int v59; // [rsp+28h] [rbp-68h] BYREF
  int v60; // [rsp+2Ch] [rbp-64h] BYREF
  __m128i *v61; // [rsp+30h] [rbp-60h] BYREF
  __m128i *v62; // [rsp+38h] [rbp-58h] BYREF
  _QWORD v63[2]; // [rsp+40h] [rbp-50h] BYREF
  int v64; // [rsp+50h] [rbp-40h]

  v7 = a1;
  v8 = (__int64 **)a2;
  v10 = *(_BYTE *)(a1 + 80);
  v11 = *(_QWORD *)(a1 + 88);
  if ( a7 )
  {
LABEL_28:
    v12 = v11;
    v13 = *(_BYTE *)(v11 + 265) & 1;
    if ( v10 != 19 )
      goto LABEL_4;
    goto LABEL_29;
  }
  if ( v10 != 19 )
  {
    v12 = *(_QWORD *)(a1 + 88);
    v13 = *(_BYTE *)(v11 + 265) & 1;
    goto LABEL_4;
  }
  if ( (*(_BYTE *)(v11 + 266) & 1) != 0 )
  {
    v7 = *(_QWORD *)(v11 + 200);
    v11 = *(_QWORD *)(v7 + 88);
    v10 = *(_BYTE *)(v7 + 80);
    goto LABEL_28;
  }
  v13 = *(_BYTE *)(v11 + 265) & 1;
LABEL_29:
  v27 = *(_QWORD *)(v11 + 200);
  v12 = v11;
  if ( v27 )
    v11 = *(_QWORD *)(v27 + 88);
LABEL_4:
  v14 = *a2;
  v15 = *(_BYTE *)(v11 + 160);
  v16 = 2 * ((v15 & 6) != 0);
  if ( (v15 & 0x10) != 0 )
    v16 = (2 * ((v15 & 6) != 0)) | 0x20;
  v57 = v16;
  sub_89A380(v13, v14, &v61, &v62, &v60, &v59);
  v18 = v57;
  if ( !v59 )
  {
    if ( unk_4D049D8 == v7 )
    {
      v13 = 1;
      v7 = unk_4D049D0;
      v12 = *(_QWORD *)(unk_4D049D0 + 88LL);
    }
    else if ( unk_4D049C8 == v7 )
    {
      v13 = 1;
      v7 = unk_4D049C0;
      v12 = *(_QWORD *)(unk_4D049C0 + 88LL);
    }
    else
    {
      v17 = a7;
      if ( !a7 && (*(_BYTE *)(v12 + 266) & 0x40) == 0 )
      {
        v36 = sub_8A00C0(v7, *v8, 1);
        v18 = v57;
        if ( !v36 )
        {
          v22 = 0;
          if ( !a6 )
          {
            v37 = *(_QWORD *)(v12 + 176);
            if ( v37 )
            {
              v38 = *(_BYTE *)(v37 + 80);
              v39 = *(_BYTE *)(*(_QWORD *)(v37 + 88) + 140LL);
            }
            else
            {
              v39 = 10;
              v38 = 4;
            }
            v22 = sub_87EBB0(v38, *(_QWORD *)v7, (_QWORD *)(v7 + 48));
            *((_DWORD *)v22 + 10) = *(_DWORD *)(v7 + 40);
            v40 = sub_7259C0(v39);
            v41 = *((_BYTE *)v22 + 80) == 3;
            v22[11] = (__int64)v40;
            v42 = (__int64)v40;
            if ( v41 )
            {
              v40[20] = sub_72C930();
            }
            else
            {
              sub_72AD80(v40);
              *(_BYTE *)(v42 + 141) &= ~0x20u;
              *(_QWORD *)(v42 + 128) = 1;
              *(_DWORD *)(v42 + 136) = 1;
              *((_BYTE *)v22 + 81) |= 2u;
            }
            sub_877D80(v42, v22);
            v46 = *(_QWORD *)(v7 + 64);
            if ( (*(_BYTE *)(v7 + 81) & 0x10) != 0 )
            {
              sub_877E20((__int64)v22, v42, v46, v43, v44, v45);
            }
            else if ( v46 )
            {
              sub_877E90((__int64)v22, v42, v46);
            }
            *((_BYTE *)v22 + 81) |= 0x20u;
            sub_725130(*v8);
          }
          goto LABEL_25;
        }
      }
    }
  }
  v19 = v62;
  v58 = a3 != 0;
  v20 = a3 != 0;
  v21 = v62;
  if ( !a4 && !a3 )
    goto LABEL_18;
  v22 = *(__int64 **)(v12 + 176);
  if ( v22 && (!*(_QWORD *)(v12 + 88) || (*(_BYTE *)(v12 + 160) & 1) != 0) && (v22 == a4 || v58) )
  {
    v48 = *((unsigned __int8 *)v22 + 80);
    v49 = v22[11];
    if ( (_BYTE)v48 == 3 )
    {
      v50 = *(_QWORD *)(*(_QWORD *)(v49 + 168) + 8LL);
    }
    else
    {
      v20 = (unsigned int)(v48 - 4);
      if ( (unsigned __int8)(v48 - 4) <= 1u )
      {
        v50 = *(_QWORD *)(*(_QWORD *)(v49 + 168) + 168LL);
      }
      else if ( (_BYTE)v48 == 7 )
      {
        v50 = **(_QWORD **)(v49 + 216);
      }
      else
      {
        v50 = *(_QWORD *)(v49 + 240);
      }
    }
    v54 = v18;
    v51 = sub_89AB40(v50, (__int64)v62, v18 | 0x48u, v20, v62);
    v18 = v54;
    if ( v51 )
    {
      if ( v13 )
      {
        if ( a6 )
          goto LABEL_22;
        goto LABEL_33;
      }
      if ( sub_890680((__int64)v22, 1) )
        goto LABEL_46;
      v19 = v62;
      v18 = v54;
      goto LABEL_36;
    }
    v19 = v62;
    if ( v13 != 1 )
      goto LABEL_36;
LABEL_51:
    v21 = v62;
LABEL_18:
    v23 = *(unsigned __int8 **)(v12 + 136);
    v63[0] = v7;
    v63[1] = v21;
    v64 = 0;
    if ( !v23 )
    {
      v47 = sub_881A70(0, 0xBu, 12, 13, (__int64)v19, v17);
      *(_QWORD *)(v12 + 136) = v47;
      v23 = (unsigned __int8 *)v47;
    }
    v24 = (__int64 *)sub_881B20(v23, (__int64)v63, 0);
    v22 = v24;
    if ( v24 )
    {
      v22 = (__int64 *)*v24;
      if ( a6 )
        goto LABEL_22;
      if ( v22 )
      {
LABEL_32:
        if ( !v13 )
          goto LABEL_22;
LABEL_33:
        if ( !*(_QWORD *)(v22[11] + 160) )
          goto LABEL_34;
LABEL_22:
        if ( v60 )
          goto LABEL_48;
        goto LABEL_23;
      }
    }
    else if ( a6 )
    {
      goto LABEL_22;
    }
    if ( !v13 )
    {
      v22 = sub_893FE0(v7, v62->m128i_i64, v59);
LABEL_58:
      v25 = v61;
      if ( v62 != v61 )
        goto LABEL_24;
      goto LABEL_25;
    }
    v22 = 0;
LABEL_34:
    v22 = (__int64 *)sub_8A09D0(v7, *v8, v62, v22);
    goto LABEL_58;
  }
  if ( v13 == 1 )
    goto LABEL_18;
LABEL_36:
  v21 = v19;
  if ( !*(_QWORD *)(v12 + 144) )
    goto LABEL_18;
  v53 = v13;
  v52 = v8;
  v28 = *(_QWORD *)(v12 + 144);
  v29 = (__int64)v19;
  v30 = v18 | 0x48;
  while ( 1 )
  {
    v31 = *(_BYTE *)(v28 + 80);
    if ( v31 == 19 )
    {
      v35 = *(__int64 **)(*(_QWORD *)(v28 + 88) + 176LL);
      if ( a4 != v35 && !v58 )
      {
LABEL_50:
        v13 = v53;
        v8 = v52;
        goto LABEL_51;
      }
      v32 = *(_QWORD *)(*(_QWORD *)(v35[11] + 168) + 168LL);
    }
    else
    {
      if ( v31 != 21 )
        sub_721090();
      v32 = **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v28 + 88) + 192LL) + 216LL);
    }
    if ( !sub_89AB40(v32, v29, v30, v20, v19) )
      goto LABEL_49;
    v33 = *(_QWORD *)(v28 + 88);
    if ( *(_BYTE *)(v28 + 80) != 19 )
      break;
    if ( sub_890680(*(_QWORD *)(v33 + 176), 0) )
    {
      v34 = v28;
      v13 = v53;
      v8 = v52;
      v33 = *(_QWORD *)(v34 + 88);
      goto LABEL_45;
    }
LABEL_49:
    v28 = *(_QWORD *)(v28 + 8);
    if ( !v28 )
      goto LABEL_50;
  }
  v13 = v53;
  v8 = v52;
LABEL_45:
  v22 = *(__int64 **)(v33 + 176);
  if ( !v22 )
    goto LABEL_51;
LABEL_46:
  if ( !a6 )
    goto LABEL_32;
  if ( v60 )
LABEL_48:
    sub_725130(v61->m128i_i64);
LABEL_23:
  v25 = (__m128i *)*v8;
LABEL_24:
  sub_725130(v25->m128i_i64);
LABEL_25:
  *v8 = 0;
  return v22;
}
