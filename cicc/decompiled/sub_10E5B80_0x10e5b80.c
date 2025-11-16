// Function: sub_10E5B80
// Address: 0x10e5b80
//
__int64 __fastcall sub_10E5B80(_QWORD *a1, __int64 a2)
{
  __int64 *v3; // r15
  __int64 v4; // rbx
  unsigned __int64 v5; // r12
  __int64 v6; // r13
  unsigned __int8 *v7; // rax
  unsigned __int8 v8; // r12
  unsigned __int16 v9; // ax
  __int64 *v10; // rax
  __int64 *v11; // rax
  __int64 v12; // rbx
  __int64 *v13; // rax
  __int64 v15; // r13
  char v16; // al
  __int64 v17; // rdx
  unsigned __int8 *v18; // r13
  __int64 v19; // rbx
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rcx
  _QWORD *v24; // r11
  int v25; // eax
  __int64 v26; // rdx
  unsigned __int16 v27; // ax
  unsigned __int8 v28; // cl
  __int64 v29; // rax
  unsigned __int64 v30; // rdx
  __int64 *v31; // rax
  __int64 v32; // rax
  _QWORD *v33; // r11
  __int64 v34; // rax
  __int64 *v35; // r15
  __int64 v36; // rax
  bool v37; // al
  __int64 v38; // rax
  char v39; // al
  __int64 v40; // r9
  __int64 v41; // rdx
  __int64 v42; // r15
  __int64 v43; // rdx
  unsigned int v44; // esi
  unsigned __int64 v45; // rsi
  __int64 v46; // rdi
  __int64 v47; // r8
  __int64 v48; // rdx
  bool v49; // al
  __int64 v50; // r8
  __int16 v51; // ax
  __int64 v52; // rdx
  __int16 v53; // ax
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 *v56; // r12
  __int64 v57; // r13
  __int64 v58; // [rsp+0h] [rbp-C0h]
  unsigned __int8 *v59; // [rsp+8h] [rbp-B8h]
  __int64 v60; // [rsp+8h] [rbp-B8h]
  char v61; // [rsp+10h] [rbp-B0h]
  unsigned __int8 v62; // [rsp+17h] [rbp-A9h]
  unsigned int v63; // [rsp+18h] [rbp-A8h]
  _QWORD *v64; // [rsp+18h] [rbp-A8h]
  int v65; // [rsp+18h] [rbp-A8h]
  _QWORD *v66; // [rsp+18h] [rbp-A8h]
  char v67; // [rsp+20h] [rbp-A0h]
  char v68; // [rsp+20h] [rbp-A0h]
  __int64 v69; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v70; // [rsp+28h] [rbp-98h]
  _QWORD *v71; // [rsp+28h] [rbp-98h]
  int v72; // [rsp+28h] [rbp-98h]
  __int64 v73; // [rsp+28h] [rbp-98h]
  __int64 v74; // [rsp+28h] [rbp-98h]
  __int64 *v75; // [rsp+28h] [rbp-98h]
  unsigned __int8 *v76; // [rsp+30h] [rbp-90h] BYREF
  unsigned __int8 *v77; // [rsp+38h] [rbp-88h]
  unsigned __int8 *v78; // [rsp+40h] [rbp-80h] BYREF
  unsigned __int8 *i; // [rsp+48h] [rbp-78h]
  __m128i v80; // [rsp+50h] [rbp-70h] BYREF
  _QWORD v81[2]; // [rsp+60h] [rbp-60h] BYREF
  __int64 v82; // [rsp+70h] [rbp-50h]
  __int64 v83; // [rsp+78h] [rbp-48h]

  v3 = (__int64 *)(a2 + 72);
  v4 = a1[8];
  v5 = a1[11];
  v6 = a1[10];
  v7 = sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), a2);
  v8 = sub_F518D0(v7, 0, v5, a2, v4, v6);
  v9 = sub_A74840((_QWORD *)(a2 + 72), 0);
  if ( !HIBYTE(v9) || v8 > (unsigned __int8)v9 )
  {
    v10 = (__int64 *)sub_BD5C60(a2);
    *(_QWORD *)(a2 + 72) = sub_A7B980(v3, v10, 1, 86);
    v11 = (__int64 *)sub_BD5C60(a2);
    v80.m128i_i32[0] = 0;
    v12 = sub_A77A40(v11, v8);
    v13 = (__int64 *)sub_BD5C60(a2);
    *(_QWORD *)(a2 + 72) = sub_A7B660(v3, v13, &v80, 1, v12);
    return a2;
  }
  v15 = a1[7];
  v80.m128i_i64[0] = (__int64)sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), 0);
  v80.m128i_i64[1] = -1;
  v81[0] = 0;
  v81[1] = 0;
  v82 = 0;
  v83 = 0;
  v16 = sub_CF5020(v15, (__int64)&v80, 0);
  v17 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( (v16 & 2) != 0 )
  {
    v18 = *(unsigned __int8 **)(a2 + 32 * (1 - v17));
    v19 = *(_QWORD *)(a2 + 32 * (2 - v17));
    if ( (unsigned int)*v18 - 12 > 1 )
    {
      v67 = *v18 != 17 || *(_BYTE *)v19 != 17;
      if ( v67 || !sub_BCAC40(*((_QWORD *)v18 + 1), 8) )
        return 0;
      v24 = a1;
      v63 = *(_DWORD *)(v19 + 32);
      if ( v63 <= 0x40 )
      {
        v26 = *(_QWORD *)(v19 + 24);
      }
      else
      {
        v25 = sub_C444A0(v19 + 24);
        v24 = a1;
        v26 = -1;
        if ( v63 - v25 <= 0x40 )
          v26 = **(_QWORD **)(v19 + 24);
      }
      v64 = v24;
      v70 = v26;
      v27 = sub_A74840(v3, 0);
      v28 = 0;
      if ( HIBYTE(v27) )
        v28 = v27;
      v29 = *(_QWORD *)(a2 - 32);
      v62 = v28;
      if ( !v29 || *(_BYTE *)v29 || *(_QWORD *)(v29 + 24) != *(_QWORD *)(a2 + 80) )
        BUG();
      v30 = v70;
      if ( *(_DWORD *)(v29 + 36) == 244 && 1LL << v28 < v70 )
        return 0;
      v71 = v64;
      if ( v30 - 1 > 7 )
        return 0;
      v65 = v30;
      if ( ((unsigned int)v30 & ((_DWORD)v30 - 1)) != 0 )
        return 0;
      v59 = sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), 0);
      sub_C47700((__int64)&v80, 8 * v65, (__int64)(v18 + 24));
      v31 = (__int64 *)sub_BD5C60(a2);
      v32 = sub_ACCFD0(v31, (__int64)&v80);
      v33 = v71;
      v58 = v32;
      if ( v80.m128i_i32[2] > 0x40u && v80.m128i_i64[0] )
      {
        j_j___libc_free_0_0(v80.m128i_i64[0]);
        v33 = v71;
      }
      v34 = *(_QWORD *)(a2 - 32);
      v35 = (__int64 *)v33[4];
      if ( !v34 || *(_BYTE *)v34 || *(_QWORD *)(v34 + 24) != *(_QWORD *)(a2 + 80) )
        BUG();
      if ( (unsigned int)(*(_DWORD *)(v34 + 36) - 238) <= 7 && ((1LL << (*(_BYTE *)(v34 + 36) + 18)) & 0xAD) != 0 )
      {
        v36 = *(_QWORD *)(a2 + 32 * (3LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
        if ( *(_DWORD *)(v36 + 32) <= 0x40u )
        {
          v37 = *(_QWORD *)(v36 + 24) == 0;
        }
        else
        {
          v72 = *(_DWORD *)(v36 + 32);
          v37 = v72 == (unsigned int)sub_C444A0(v36 + 24);
        }
        v67 = !v37;
      }
      v61 = v67;
      v38 = sub_AA4E30(v35[6]);
      v39 = sub_AE5020(v38, *(_QWORD *)(v58 + 8));
      LOWORD(v82) = 257;
      v68 = v39;
      v66 = sub_BD2C40(80, unk_3F10A10);
      if ( v66 )
        sub_B4D3C0((__int64)v66, v58, (__int64)v59, v61, v68, v40, 0, 0);
      (*(void (__fastcall **)(__int64, _QWORD *, __m128i *, __int64, __int64))(*(_QWORD *)v35[11] + 16LL))(
        v35[11],
        v66,
        &v80,
        v35[7],
        v35[8]);
      v41 = 16LL * *((unsigned int *)v35 + 2);
      v42 = *v35;
      v73 = v42 + v41;
      while ( v73 != v42 )
      {
        v43 = *(_QWORD *)(v42 + 8);
        v44 = *(_DWORD *)v42;
        v42 += 16;
        sub_B99FD0((__int64)v66, v44, v43);
      }
      v45 = a2;
      v80.m128i_i32[0] = 38;
      sub_B47C00((__int64)v66, a2, v80.m128i_i32, 1);
      if ( (*((_BYTE *)v66 + 7) & 0x20) != 0 )
      {
        v45 = 38;
        v46 = sub_B91C10((__int64)v66, 38);
        if ( v46 )
        {
          v78 = v18;
          v47 = sub_AE94B0(v46);
          v60 = v48;
          for ( i = (unsigned __int8 *)v58; v47 != v60; v47 = *(_QWORD *)(v50 + 8) )
          {
            v69 = v47;
            v74 = *(_QWORD *)(v47 + 24);
            sub_B58E30(&v80, v74);
            v45 = (unsigned __int64)&v78;
            v49 = sub_10E5A70(v80.m128i_i64, (__int64 *)&v78);
            v50 = v69;
            if ( v49 )
            {
              v45 = (unsigned __int64)v78;
              sub_B59720(v74, (__int64)v78, i);
              v50 = v69;
            }
          }
        }
        if ( (*((_BYTE *)v66 + 7) & 0x20) != 0 )
        {
          v45 = 38;
          v54 = sub_B91C10((__int64)v66, 38);
          if ( v54 )
          {
            v55 = *(_QWORD *)(v54 + 8);
            v45 = v55 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (v55 & 4) == 0 )
              v45 = 0;
            sub_B967C0(&v80, (__m128i *)v45);
            v56 = (__int64 *)v80.m128i_i64[0];
            v76 = v18;
            v75 = (__int64 *)(v80.m128i_i64[0] + 8LL * v80.m128i_u32[2]);
            v77 = (unsigned __int8 *)v58;
            if ( (__int64 *)v80.m128i_i64[0] != v75 )
            {
              do
              {
                v57 = *v56;
                sub_B129C0(&v78, *v56);
                v45 = (unsigned __int64)&v76;
                if ( sub_10E5AF0((__int64 *)&v78, (__int64 *)&v76) )
                {
                  v45 = (unsigned __int64)v76;
                  sub_B13360(v57, v76, v77, 0);
                }
                ++v56;
              }
              while ( v75 != v56 );
              v75 = (__int64 *)v80.m128i_i64[0];
            }
            if ( v75 != v81 )
              _libc_free(v75, v45);
          }
        }
      }
      v51 = *((_WORD *)v66 + 1) & 0xFF81 | (2 * v62);
      *((_WORD *)v66 + 1) = v51;
      v52 = *(_QWORD *)(a2 - 32);
      if ( !v52 || *(_BYTE *)v52 || *(_QWORD *)(v52 + 24) != *(_QWORD *)(a2 + 80) )
        BUG();
      if ( *(_DWORD *)(v52 + 36) == 244 )
      {
        v53 = v51 & 0xFC7F;
        LOBYTE(v53) = v53 | 0x80;
        *((_WORD *)v66 + 1) = v53;
      }
      v20 = sub_AD6530(*(_QWORD *)(v19 + 8), v45);
      v21 = a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
      if ( *(_QWORD *)v21 )
        goto LABEL_7;
    }
    else
    {
      v20 = sub_AD6530(*(_QWORD *)(v19 + 8), 2);
      v21 = a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
      if ( *(_QWORD *)v21 )
        goto LABEL_7;
    }
  }
  else
  {
    v20 = sub_AD6530(*(_QWORD *)(*(_QWORD *)(a2 + 32 * (2 - v17)) + 8LL), (__int64)&v80);
    v21 = a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    if ( *(_QWORD *)v21 )
    {
LABEL_7:
      v22 = *(_QWORD *)(v21 + 8);
      **(_QWORD **)(v21 + 16) = v22;
      if ( v22 )
        *(_QWORD *)(v22 + 16) = *(_QWORD *)(v21 + 16);
    }
  }
  *(_QWORD *)v21 = v20;
  if ( v20 )
  {
    v23 = *(_QWORD *)(v20 + 16);
    *(_QWORD *)(v21 + 8) = v23;
    if ( v23 )
      *(_QWORD *)(v23 + 16) = v21 + 8;
    *(_QWORD *)(v21 + 16) = v20 + 16;
    *(_QWORD *)(v20 + 16) = v21;
  }
  return a2;
}
