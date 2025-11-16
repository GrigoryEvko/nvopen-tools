// Function: sub_10EE020
// Address: 0x10ee020
//
__int64 __fastcall sub_10EE020(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r15
  __int64 v4; // r12
  __int64 v5; // r14
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rbx
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rbx
  __int64 v13; // rcx
  __int64 v14; // rcx
  __int64 v15; // r13
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v19; // rcx
  __int64 v20; // rdx
  unsigned __int64 v21; // rax
  __int64 v22; // r15
  __int64 v23; // rax
  bool v24; // zf
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 *v27; // rbx
  __int64 v28; // r15
  __int64 v29; // rax
  __int64 v30; // r15
  int v31; // eax
  __int64 *v32; // r11
  __int64 v33; // r13
  __int64 v34; // rbx
  __int64 v35; // r9
  _QWORD *v36; // r13
  __int64 v37; // rdx
  unsigned __int64 v38; // rax
  char v39; // r15
  __int64 v40; // r9
  __int64 v41; // rbx
  __int64 v42; // rax
  __int64 v43; // r15
  __int64 v44; // rdx
  __int64 v45; // rdx
  __int64 v46; // r13
  _QWORD *v47; // rax
  __int64 *v48; // r11
  __int64 v49; // r13
  __int64 v50; // r15
  __int64 v51; // rdx
  unsigned int v52; // esi
  __int64 v53; // rdx
  __int64 v54; // rbx
  __int64 v55; // rdx
  unsigned int v56; // esi
  __int64 v57; // [rsp+8h] [rbp-B8h]
  char v58; // [rsp+17h] [rbp-A9h]
  __int64 v59; // [rsp+18h] [rbp-A8h]
  __int64 v60; // [rsp+18h] [rbp-A8h]
  __int64 *v61; // [rsp+18h] [rbp-A8h]
  __int64 *v62; // [rsp+18h] [rbp-A8h]
  __int64 v63; // [rsp+18h] [rbp-A8h]
  __int64 v64; // [rsp+20h] [rbp-A0h] BYREF
  unsigned int v65; // [rsp+28h] [rbp-98h]
  __int64 v66; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v67; // [rsp+38h] [rbp-88h]
  __int16 v68; // [rsp+50h] [rbp-70h]
  __int64 v69; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v70; // [rsp+68h] [rbp-58h]
  __int16 v71; // [rsp+80h] [rbp-40h]

  v2 = 32 * (3LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v3 = *(_QWORD *)(a2 + v2);
  if ( *(_BYTE *)v3 <= 0x15u )
  {
    v4 = a2;
    if ( sub_AC30F0(*(_QWORD *)(a2 + v2)) )
      return sub_F207A0(a1, (__int64 *)a2);
    v5 = sub_9B7920(*(char **)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
    if ( v5 )
    {
      v8 = sub_9B7920(*(char **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
      if ( v8 && (unsigned __int8)sub_9BA100((unsigned __int8 *)v3) )
      {
        v37 = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
        v38 = *(_QWORD *)(v37 + 24);
        if ( *(_DWORD *)(v37 + 32) > 0x40u )
          v38 = *(_QWORD *)v38;
        v39 = 0;
        if ( v38 )
        {
          _BitScanReverse64(&v38, v38);
          v39 = 63 - (v38 ^ 0x3F);
        }
        v36 = sub_BD2C40(80, unk_3F10A10);
        if ( v36 )
          sub_B4D3C0((__int64)v36, v8, v5, 0, v39, v40, 0, 0);
      }
      else
      {
        if ( !sub_AD7930((_BYTE *)v3, a2, v6, v7, v9) )
          goto LABEL_7;
        v19 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
        v20 = *(_QWORD *)(a2 + 32 * (2 - v19));
        v21 = *(_QWORD *)(v20 + 24);
        if ( *(_DWORD *)(v20 + 32) > 0x40u )
          v21 = *(_QWORD *)v21;
        v58 = 0;
        if ( v21 )
        {
          _BitScanReverse64(&v21, v21);
          v58 = 63 - (v21 ^ 0x3F);
        }
        v22 = *(_QWORD *)(a1 + 32);
        v23 = *(_QWORD *)(*(_QWORD *)(a2 + 32 * (1 - v19)) + 8LL);
        v24 = *(_BYTE *)(v23 + 8) == 18;
        LODWORD(v23) = *(_DWORD *)(v23 + 32);
        BYTE4(v64) = v24;
        LODWORD(v64) = v23;
        v25 = sub_BCB2D0(*(_QWORD **)(v22 + 72));
        v26 = sub_B33F10(v22, v25, v64);
        v27 = *(__int64 **)(a1 + 32);
        v68 = 257;
        v28 = v26;
        v57 = v26;
        v29 = sub_BCB2D0((_QWORD *)v27[9]);
        v60 = sub_ACD640(v29, 1, 0);
        v30 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v27[10] + 32LL))(
                v27[10],
                15,
                v28,
                v60,
                0,
                0);
        if ( !v30 )
        {
          v71 = 257;
          v30 = sub_B504D0(15, v57, v60, (__int64)&v69, 0, 0);
          (*(void (__fastcall **)(__int64, __int64, __int64 *, __int64, __int64))(*(_QWORD *)v27[11] + 16LL))(
            v27[11],
            v30,
            &v66,
            v27[7],
            v27[8]);
          v53 = 16LL * *((unsigned int *)v27 + 2);
          v54 = *v27;
          v63 = v54 + v53;
          while ( v63 != v54 )
          {
            v55 = *(_QWORD *)(v54 + 8);
            v56 = *(_DWORD *)v54;
            v54 += 16;
            sub_B99FD0(v30, v56, v55);
          }
        }
        v31 = *(_DWORD *)(v4 + 4);
        v32 = *(__int64 **)(a1 + 32);
        v68 = 257;
        v61 = v32;
        v33 = *(_QWORD *)(v4 - 32LL * (v31 & 0x7FFFFFF));
        v34 = (*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v32[10] + 96LL))(v32[10], v33, v30);
        if ( !v34 )
        {
          v71 = 257;
          v47 = sub_BD2C40(72, 2u);
          v48 = v61;
          v34 = (__int64)v47;
          if ( v47 )
          {
            sub_B4DE80((__int64)v47, v33, v30, (__int64)&v69, 0, 0);
            v48 = v61;
          }
          v62 = v48;
          (*(void (__fastcall **)(__int64, __int64, __int64 *, __int64, __int64))(*(_QWORD *)v48[11] + 16LL))(
            v48[11],
            v34,
            &v66,
            v48[7],
            v48[8]);
          v49 = *v62;
          v50 = *v62 + 16LL * *((unsigned int *)v62 + 2);
          if ( *v62 != v50 )
          {
            do
            {
              v51 = *(_QWORD *)(v49 + 8);
              v52 = *(_DWORD *)v49;
              v49 += 16;
              sub_B99FD0(v34, v52, v51);
            }
            while ( v50 != v49 );
          }
        }
        v36 = sub_BD2C40(80, unk_3F10A10);
        if ( v36 )
          sub_B4D3C0((__int64)v36, v34, v5, 0, v58, v35, 0, 0);
      }
      sub_B47C00((__int64)v36, v4, 0, 0);
      return (__int64)v36;
    }
LABEL_7:
    if ( *(_BYTE *)(*(_QWORD *)(v3 + 8) + 8LL) != 18 )
    {
      sub_9BA1B0(&v64, v3);
      v67 = v65;
      if ( v65 > 0x40 )
      {
        sub_C43690((__int64)&v66, 0, 0);
        v70 = v65;
        if ( v65 > 0x40 )
        {
          sub_C43780((__int64)&v69, (const void **)&v64);
          goto LABEL_11;
        }
      }
      else
      {
        v70 = v65;
        v66 = 0;
      }
      v69 = v64;
LABEL_11:
      v10 = sub_11A3F30(a1, *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), &v69, &v66, 0, 0);
      if ( v70 > 0x40 && v69 )
      {
        v59 = v10;
        j_j___libc_free_0_0(v69);
        v10 = v59;
      }
      if ( v10 )
      {
        if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
          v11 = *(_QWORD *)(a2 - 8);
        else
          v11 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
        v12 = *(_QWORD *)v11;
        if ( *(_QWORD *)v11 )
        {
          v13 = *(_QWORD *)(v11 + 8);
          **(_QWORD **)(v11 + 16) = v13;
          if ( v13 )
            *(_QWORD *)(v13 + 16) = *(_QWORD *)(v11 + 16);
        }
        *(_QWORD *)v11 = v10;
        v14 = *(_QWORD *)(v10 + 16);
        *(_QWORD *)(v11 + 8) = v14;
        if ( v14 )
          *(_QWORD *)(v14 + 16) = v11 + 8;
        *(_QWORD *)(v11 + 16) = v10 + 16;
        *(_QWORD *)(v10 + 16) = v11;
        if ( *(_BYTE *)v12 <= 0x1Cu )
          goto LABEL_24;
        v15 = *(_QWORD *)(a1 + 40);
        v69 = v12;
        v16 = v15 + 2096;
        sub_10E8740(v16, &v69);
        v17 = *(_QWORD *)(v12 + 16);
        if ( !v17 )
          goto LABEL_24;
      }
      else
      {
        v70 = v65;
        if ( v65 > 0x40 )
          sub_C43780((__int64)&v69, (const void **)&v64);
        else
          v69 = v64;
        v41 = sub_11A3F30(a1, *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))), &v69, &v66, 0, 0);
        if ( v70 > 0x40 && v69 )
          j_j___libc_free_0_0(v69);
        if ( !v41 )
        {
          v4 = 0;
LABEL_24:
          if ( v67 > 0x40 && v66 )
            j_j___libc_free_0_0(v66);
          if ( v65 > 0x40 )
          {
            if ( v64 )
              j_j___libc_free_0_0(v64);
          }
          return v4;
        }
        if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
          v42 = *(_QWORD *)(a2 - 8);
        else
          v42 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
        v43 = *(_QWORD *)(v42 + 32);
        if ( v43 )
        {
          v44 = *(_QWORD *)(v42 + 40);
          **(_QWORD **)(v42 + 48) = v44;
          if ( v44 )
            *(_QWORD *)(v44 + 16) = *(_QWORD *)(v42 + 48);
        }
        *(_QWORD *)(v42 + 32) = v41;
        v45 = *(_QWORD *)(v41 + 16);
        *(_QWORD *)(v42 + 40) = v45;
        if ( v45 )
          *(_QWORD *)(v45 + 16) = v42 + 40;
        *(_QWORD *)(v42 + 48) = v41 + 16;
        *(_QWORD *)(v41 + 16) = v42 + 32;
        if ( *(_BYTE *)v43 <= 0x1Cu )
          goto LABEL_24;
        v46 = *(_QWORD *)(a1 + 40);
        v69 = v43;
        v16 = v46 + 2096;
        sub_10E8740(v16, &v69);
        v17 = *(_QWORD *)(v43 + 16);
        if ( !v17 )
          goto LABEL_24;
      }
      if ( !*(_QWORD *)(v17 + 8) )
      {
        v69 = *(_QWORD *)(v17 + 24);
        sub_10E8740(v16, &v69);
      }
      goto LABEL_24;
    }
  }
  return 0;
}
