// Function: sub_26330D0
// Address: 0x26330d0
//
void __fastcall sub_26330D0(__int64 **a1, __int64 a2, unsigned __int8 a3, __int64 a4)
{
  unsigned __int8 v6; // bl
  char *v7; // rax
  char v8; // bl
  __int64 v9; // rdx
  char v10; // al
  __int64 v11; // rsi
  unsigned int v12; // r15d
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // r9
  unsigned int v16; // r15d
  __int64 v17; // r14
  char v18; // bl
  __int64 v19; // r14
  __int64 v20; // rsi
  __int64 *v21; // r9
  __int64 v22; // rax
  unsigned __int8 *v23; // r15
  unsigned __int8 **v24; // rsi
  char v25; // bl
  __int64 *v26; // r15
  __int64 v27; // rax
  __int64 v28; // rax
  char v29; // al
  __int64 *v30; // rbx
  __int64 v31; // rax
  __int64 v32; // rsi
  unsigned int v33; // r15d
  __int64 v34; // rax
  unsigned int v35; // r15d
  __int64 v36; // r13
  __int64 v37; // [rsp+0h] [rbp-120h]
  __int64 *v39; // [rsp+20h] [rbp-100h]
  __int64 v40; // [rsp+20h] [rbp-100h]
  unsigned int v41; // [rsp+28h] [rbp-F8h]
  unsigned int v42; // [rsp+28h] [rbp-F8h]
  __int64 v43; // [rsp+30h] [rbp-F0h]
  _BYTE *v44; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v45; // [rsp+48h] [rbp-D8h]
  _QWORD v46[2]; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v47[2]; // [rsp+60h] [rbp-C0h] BYREF
  char v48; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v49[2]; // [rsp+80h] [rbp-A0h] BYREF
  _QWORD v50[2]; // [rsp+90h] [rbp-90h] BYREF
  unsigned __int8 *v51[2]; // [rsp+A0h] [rbp-80h] BYREF
  _QWORD v52[2]; // [rsp+B0h] [rbp-70h] BYREF
  const char *v53[4]; // [rsp+C0h] [rbp-60h] BYREF
  __int16 v54; // [rsp+E0h] [rbp-40h]

  v6 = *(_BYTE *)(a2 + 32);
  v7 = (char *)sub_BD5D20(a2);
  v8 = (v6 >> 4) & 3;
  v44 = v46;
  sub_2619AF0((__int64 *)&v44, v7, (__int64)&v7[v9]);
  if ( (*(_BYTE *)(a2 + 32) & 0xF) != 1 && !sub_B2FC80(a2) )
  {
    if ( a3 )
    {
      v51[0] = (unsigned __int8 *)v52;
      sub_261A960((__int64 *)v51, v44, (__int64)&v44[v45]);
      if ( 0x3FFFFFFFFFFFFFFFLL - (unsigned __int64)v51[1] > 3 )
      {
        sub_2241490((unsigned __int64 *)v51, ".cfi", 4u);
        v54 = 260;
        v53[0] = (const char *)v51;
        sub_BD6B50((unsigned __int8 *)a2, v53);
        if ( (_QWORD *)v51[0] != v52 )
          j_j___libc_free_0((unsigned __int64)v51[0]);
        v10 = *(_BYTE *)(a2 + 32);
        *(_BYTE *)(a2 + 32) = v10 & 0xF0;
        if ( (v10 & 0x30) != 0 )
          *(_BYTE *)(a2 + 33) |= 0x40u;
        v11 = *(_QWORD *)(a2 + 24);
        v53[0] = (const char *)&v44;
        v54 = 260;
        v12 = *(_DWORD *)(*(_QWORD *)(a2 + 8) + 8LL);
        v39 = *a1;
        v13 = sub_BD2DA0(136);
        v16 = v12 >> 8;
        v17 = v13;
        if ( v13 )
          sub_B2C3B0(v13, v11, 0, v16, (__int64)v53, (__int64)v39);
        v18 = (16 * (v8 & 3)) | *(_BYTE *)(v17 + 32) & 0xCF;
        *(_BYTE *)(v17 + 32) = v18;
        if ( (v18 & 0xFu) - 7 <= 1 || (v18 & 0x30) != 0 && (v18 & 0xF) != 9 )
          *(_BYTE *)(v17 + 33) |= 0x40u;
        if ( *(_QWORD *)(a2 + 16) )
        {
          v37 = v17;
          v19 = *(_QWORD *)(a2 + 16);
          do
          {
            if ( **(_BYTE **)(v19 + 24) == 1 )
            {
              v51[0] = *(unsigned __int8 **)(v19 + 24);
              v20 = *(_QWORD *)(a2 + 24);
              v21 = *a1;
              v54 = 257;
              v40 = (__int64)v21;
              v41 = *(_DWORD *)(*(_QWORD *)(a2 + 8) + 8LL) >> 8;
              v22 = sub_BD2DA0(136);
              v23 = (unsigned __int8 *)v22;
              if ( v22 )
                sub_B2C3B0(v22, v20, 0, v41, (__int64)v53, v40);
              sub_BD6B90(v23, v51[0]);
              sub_BD84D0((__int64)v51[0], (__int64)v23);
              v24 = *(unsigned __int8 ***)(a4 + 8);
              if ( v24 == *(unsigned __int8 ***)(a4 + 16) )
              {
                sub_2628AD0(a4, v24, v51);
              }
              else
              {
                if ( v24 )
                {
                  *v24 = v51[0];
                  v24 = *(unsigned __int8 ***)(a4 + 8);
                }
                *(_QWORD *)(a4 + 8) = v24 + 1;
              }
            }
            v19 = *(_QWORD *)(v19 + 8);
          }
          while ( v19 );
          v17 = v37;
          v8 = 1;
        }
        else
        {
          v8 = 1;
        }
LABEL_24:
        if ( (*(_BYTE *)(a2 + 32) & 0xF) == 9 )
          sub_2632600(a1, a2, v17, a3, v14, v15);
        else
          sub_2631F50((__int64)a1, a2, v17, a3, v14, v15);
        v25 = *(_BYTE *)(a2 + 32) & 0xCF | (16 * (v8 & 3));
        *(_BYTE *)(a2 + 32) = v25;
        if ( (v25 & 0xFu) - 7 <= 1 || (v25 & 0x30) != 0 && (v25 & 0xF) != 9 )
          *(_BYTE *)(a2 + 33) |= 0x40u;
        goto LABEL_28;
      }
      goto LABEL_55;
    }
LABEL_32:
    v49[0] = (__int64)v50;
    v26 = *a1;
    sub_261A960(v49, v44, (__int64)&v44[v45]);
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v49[1]) > 6 )
    {
      sub_2241490((unsigned __int64 *)v49, ".cfi_jt", 7u);
      v27 = *(_QWORD *)(a2 + 8);
      v53[0] = (const char *)v49;
      v54 = 260;
      v43 = *(_QWORD *)(a2 + 24);
      v42 = *(_DWORD *)(v27 + 8) >> 8;
      v28 = sub_BD2DA0(136);
      v17 = v28;
      if ( v28 )
        sub_B2C3B0(v28, v43, 0, v42, (__int64)v53, (__int64)v26);
      if ( (_QWORD *)v49[0] != v50 )
        j_j___libc_free_0(v49[0]);
      v29 = *(_BYTE *)(v17 + 32) & 0xCF | 0x10;
      *(_BYTE *)(v17 + 32) = v29;
      if ( (v29 & 0xF) != 9 )
        *(_BYTE *)(v17 + 33) |= 0x40u;
      goto LABEL_24;
    }
LABEL_55:
    sub_4262D8((__int64)"basic_string::append");
  }
  if ( !a3 )
    goto LABEL_32;
  if ( (*(_BYTE *)(a2 + 33) & 0x40) == 0 )
    goto LABEL_28;
  v30 = *a1;
  v47[0] = (__int64)&v48;
  sub_261A960(v47, v44, (__int64)&v44[v45]);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v47[1]) <= 3 )
    goto LABEL_55;
  sub_2241490((unsigned __int64 *)v47, ".cfi", 4u);
  v31 = *(_QWORD *)(a2 + 8);
  v32 = *(_QWORD *)(a2 + 24);
  v54 = 260;
  v53[0] = (const char *)v47;
  v33 = *(_DWORD *)(v31 + 8);
  v34 = sub_BD2DA0(136);
  v35 = v33 >> 8;
  v36 = v34;
  if ( v34 )
    sub_B2C3B0(v34, v32, 0, v35, (__int64)v53, (__int64)v30);
  sub_2240A30((unsigned __int64 *)v47);
  *(_BYTE *)(v36 + 32) = *(_BYTE *)(v36 + 32) & 0xCF | 0x10;
  if ( (unsigned __int8)sub_2624ED0(v36) )
    *(_BYTE *)(v36 + 33) |= 0x40u;
  sub_BD79D0(
    (__int64 *)a2,
    (__int64 *)v36,
    (unsigned __int8 (__fastcall *)(__int64, __int64 *))sub_2618A90,
    (__int64)sub_2618A70);
LABEL_28:
  if ( v44 != (_BYTE *)v46 )
    j_j___libc_free_0((unsigned __int64)v44);
}
