// Function: sub_27E1CC0
// Address: 0x27e1cc0
//
__int64 __fastcall sub_27E1CC0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r13
  _BYTE *v7; // r14
  unsigned __int16 v8; // ax
  _QWORD *v9; // rax
  _QWORD *v10; // rdx
  char v11; // cl
  _QWORD *v12; // rax
  _QWORD *v13; // rdx
  char v14; // cl
  __int64 v15; // rax
  __int64 v16; // r13
  unsigned __int8 *v17; // r13
  unsigned int v18; // r13d
  unsigned int v20; // eax
  char *v21; // rdi
  __int64 v22; // rax
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // r12
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  unsigned __int8 *v28; // rbx
  __int16 v29; // dx
  __int64 v30; // rax
  char v31; // dl
  _QWORD *v32; // r13
  const char **v33; // r8
  _QWORD *v34; // r12
  __int64 v35; // rax
  __int64 v36; // r15
  _QWORD *v37; // r14
  __int64 v38; // rdx
  int v39; // eax
  int v40; // eax
  unsigned int v41; // ecx
  __int64 v42; // rax
  __int64 v43; // rcx
  __int64 v44; // rcx
  __int64 v45; // rdx
  int v46; // eax
  int v47; // eax
  unsigned int v48; // ecx
  __int64 v49; // rax
  __int64 v50; // rcx
  __int64 v51; // rcx
  const char *v52; // rsi
  __int64 v53; // rsi
  unsigned __int8 *v54; // rsi
  __int64 v55; // [rsp+8h] [rbp-178h]
  __int64 v56; // [rsp+10h] [rbp-170h]
  char *v57; // [rsp+18h] [rbp-168h]
  __int64 v58; // [rsp+18h] [rbp-168h]
  __int64 v59; // [rsp+20h] [rbp-160h]
  char *v60; // [rsp+38h] [rbp-148h]
  __int64 v61; // [rsp+38h] [rbp-148h]
  const char **v62; // [rsp+38h] [rbp-148h]
  __int64 v63; // [rsp+38h] [rbp-148h]
  __int64 v64; // [rsp+38h] [rbp-148h]
  __int64 v65; // [rsp+48h] [rbp-138h]
  _QWORD *v66; // [rsp+48h] [rbp-138h]
  const char *v67[4]; // [rsp+50h] [rbp-130h] BYREF
  __int16 v68; // [rsp+70h] [rbp-110h]
  _QWORD *v69; // [rsp+80h] [rbp-100h] BYREF
  __int64 v70; // [rsp+88h] [rbp-F8h]
  _QWORD v71[2]; // [rsp+90h] [rbp-F0h] BYREF
  __int64 j; // [rsp+A0h] [rbp-E0h]
  _QWORD v73[3]; // [rsp+B0h] [rbp-D0h] BYREF
  int v74; // [rsp+C8h] [rbp-B8h]
  char v75; // [rsp+F0h] [rbp-90h]
  __int64 v76; // [rsp+100h] [rbp-80h] BYREF
  __int64 v77; // [rsp+108h] [rbp-78h]
  __int64 v78; // [rsp+110h] [rbp-70h]
  __int64 v79; // [rsp+118h] [rbp-68h]
  __int64 i; // [rsp+120h] [rbp-60h]
  char v81; // [rsp+140h] [rbp-40h]

  v6 = *(_QWORD *)(a4 - 96);
  v7 = *(_BYTE **)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
  v57 = *(char **)(a4 - 32);
  v60 = *(char **)(a4 - 64);
  v65 = sub_AA4E30((__int64)a2);
  v8 = sub_9A18B0(v6, v7, v65, 1u, 0);
  if ( HIBYTE(v8) && (_BYTE)v8 )
    goto LABEL_3;
  LOWORD(v20) = sub_9A18B0(v6, v7, v65, 0, 0);
  v18 = BYTE1(v20);
  if ( BYTE1(v20) )
  {
    v18 = v20;
    if ( (_BYTE)v20 )
    {
      v21 = v57;
      v57 = v60;
      v60 = v21;
LABEL_3:
      v73[0] = 0;
      v74 = 128;
      v9 = (_QWORD *)sub_C7D670(0x2000, 8);
      v73[2] = 0;
      v73[1] = v9;
      v77 = 2;
      v10 = v9 + 1024;
      v78 = 0;
      v79 = -4096;
      for ( i = 0; v10 != v9; v9 += 8 )
      {
        if ( v9 )
        {
          v11 = v77;
          v9[2] = 0;
          v9[3] = -4096;
          *v9 = &unk_49DD7B0;
          v9[1] = v11 & 6;
          v9[4] = i;
        }
      }
      v75 = 0;
      v76 = 0;
      LODWORD(v79) = 128;
      v12 = (_QWORD *)sub_C7D670(0x2000, 8);
      v78 = 0;
      v77 = (__int64)v12;
      v70 = 2;
      v13 = v12 + 1024;
      v69 = &unk_49DD7B0;
      v71[0] = 0;
      v71[1] = -4096;
      for ( j = 0; v13 != v12; v12 += 8 )
      {
        if ( v12 )
        {
          v14 = v70;
          v12[2] = 0;
          v12[3] = -4096;
          *v12 = &unk_49DD7B0;
          v12[1] = v14 & 6;
          v12[4] = j;
        }
      }
      v15 = *(_QWORD *)(a3 + 40);
      v16 = *(_QWORD *)(a3 + 32);
      v81 = 0;
      if ( v16 == v15 + 48 || !v16 )
        v17 = 0;
      else
        v17 = (unsigned __int8 *)(v16 - 24);
      if ( *(_DWORD *)(a1 + 416) < (unsigned int)sub_27DC180(*(__int64 ***)(a1 + 24), a2, v17, *(_DWORD *)(a1 + 416)) )
      {
        v18 = 0;
LABEL_16:
        sub_27DCE00((__int64)&v76);
        sub_27DCE00((__int64)v73);
        return v18;
      }
      v59 = sub_F4AB30((__int64)a2, v60, v17, (__int64)&v76, *(_QWORD *)(a1 + 48));
      v22 = sub_F4AB30((__int64)a2, v57, (_BYTE *)a3, (__int64)v73, *(_QWORD *)(a1 + 48));
      v25 = a2[7];
      v58 = v22;
      v69 = v71;
      v70 = 0x400000000LL;
      while ( 1 )
      {
        v28 = (unsigned __int8 *)(v25 - 24);
        if ( !v25 )
          v28 = 0;
        if ( v28 == v17 )
          break;
        if ( *v28 != 84 )
        {
          v26 = (unsigned int)v70;
          v27 = (unsigned int)v70 + 1LL;
          if ( v27 > HIDWORD(v70) )
          {
            sub_C8D5F0((__int64)&v69, v71, v27, 8u, v23, v24);
            v26 = (unsigned int)v70;
          }
          v69[v26] = v28;
          LODWORD(v70) = v70 + 1;
        }
        v25 = *(_QWORD *)(v25 + 8);
      }
      v56 = sub_AA5190((__int64)a2);
      if ( v56 )
      {
        LOBYTE(v30) = v29;
        v31 = HIBYTE(v29);
      }
      else
      {
        v31 = 0;
        LOBYTE(v30) = 0;
      }
      v30 = (unsigned __int8)v30;
      v66 = v69;
      BYTE1(v30) = v31;
      v32 = &v69[(unsigned int)v70];
      if ( v69 == v32 )
      {
LABEL_67:
        if ( v32 != v71 )
          _libc_free((unsigned __int64)v32);
        v18 = 1;
        goto LABEL_16;
      }
      v55 = v30;
      while ( 1 )
      {
        v34 = (_QWORD *)*(v32 - 1);
        if ( !v34[2] )
          goto LABEL_38;
        v68 = 257;
        v61 = v34[1];
        v35 = sub_BD2DA0(80);
        v36 = v35;
        if ( v35 )
        {
          v37 = (_QWORD *)v35;
          sub_B44260(v35, v61, 55, 0x8000000u, 0, 0);
          *(_DWORD *)(v36 + 72) = 2;
          sub_BD6B50((unsigned __int8 *)v36, v67);
          sub_BD2A10(v36, *(_DWORD *)(v36 + 72), 1);
        }
        else
        {
          v37 = 0;
        }
        v38 = sub_27E1A50((__int64)v73, (__int64)v34)[2];
        v39 = *(_DWORD *)(v36 + 4) & 0x7FFFFFF;
        if ( v39 == *(_DWORD *)(v36 + 72) )
        {
          v64 = v38;
          sub_B48D90(v36);
          v38 = v64;
          v39 = *(_DWORD *)(v36 + 4) & 0x7FFFFFF;
        }
        v40 = (v39 + 1) & 0x7FFFFFF;
        v41 = v40 | *(_DWORD *)(v36 + 4) & 0xF8000000;
        v42 = *(_QWORD *)(v36 - 8) + 32LL * (unsigned int)(v40 - 1);
        *(_DWORD *)(v36 + 4) = v41;
        if ( *(_QWORD *)v42 )
        {
          v43 = *(_QWORD *)(v42 + 8);
          **(_QWORD **)(v42 + 16) = v43;
          if ( v43 )
            *(_QWORD *)(v43 + 16) = *(_QWORD *)(v42 + 16);
        }
        *(_QWORD *)v42 = v38;
        if ( v38 )
        {
          v44 = *(_QWORD *)(v38 + 16);
          *(_QWORD *)(v42 + 8) = v44;
          if ( v44 )
            *(_QWORD *)(v44 + 16) = v42 + 8;
          *(_QWORD *)(v42 + 16) = v38 + 16;
          *(_QWORD *)(v38 + 16) = v42;
        }
        *(_QWORD *)(*(_QWORD *)(v36 - 8)
                  + 32LL * *(unsigned int *)(v36 + 72)
                  + 8LL * ((*(_DWORD *)(v36 + 4) & 0x7FFFFFFu) - 1)) = v58;
        v45 = sub_27E1A50((__int64)&v76, (__int64)v34)[2];
        v46 = *(_DWORD *)(v36 + 4) & 0x7FFFFFF;
        if ( v46 == *(_DWORD *)(v36 + 72) )
        {
          v63 = v45;
          sub_B48D90(v36);
          v45 = v63;
          v46 = *(_DWORD *)(v36 + 4) & 0x7FFFFFF;
        }
        v47 = (v46 + 1) & 0x7FFFFFF;
        v48 = v47 | *(_DWORD *)(v36 + 4) & 0xF8000000;
        v49 = *(_QWORD *)(v36 - 8) + 32LL * (unsigned int)(v47 - 1);
        *(_DWORD *)(v36 + 4) = v48;
        if ( *(_QWORD *)v49 )
        {
          v50 = *(_QWORD *)(v49 + 8);
          **(_QWORD **)(v49 + 16) = v50;
          if ( v50 )
            *(_QWORD *)(v50 + 16) = *(_QWORD *)(v49 + 16);
        }
        *(_QWORD *)v49 = v45;
        if ( v45 )
        {
          v51 = *(_QWORD *)(v45 + 16);
          *(_QWORD *)(v49 + 8) = v51;
          if ( v51 )
            *(_QWORD *)(v51 + 16) = v49 + 8;
          *(_QWORD *)(v49 + 16) = v45 + 16;
          *(_QWORD *)(v45 + 16) = v49;
        }
        v33 = (const char **)(v36 + 48);
        *(_QWORD *)(*(_QWORD *)(v36 - 8)
                  + 32LL * *(unsigned int *)(v36 + 72)
                  + 8LL * ((*(_DWORD *)(v36 + 4) & 0x7FFFFFFu) - 1)) = v59;
        v52 = (const char *)v34[6];
        v67[0] = v52;
        if ( v52 )
          break;
        if ( v33 != v67 )
        {
          v53 = *(_QWORD *)(v36 + 48);
          if ( v53 )
            goto LABEL_63;
        }
LABEL_37:
        sub_B44220(v37, v56, v55);
        sub_BD84D0((__int64)v34, v36);
LABEL_38:
        --v32;
        sub_B44570((__int64)v34);
        sub_B43D60(v34);
        if ( v66 == v32 )
        {
          v32 = v69;
          goto LABEL_67;
        }
      }
      sub_B96E90((__int64)v67, (__int64)v52, 1);
      v33 = (const char **)(v36 + 48);
      if ( (const char **)(v36 + 48) == v67 )
      {
        if ( v67[0] )
          sub_B91220((__int64)v67, (__int64)v67[0]);
        goto LABEL_37;
      }
      v53 = *(_QWORD *)(v36 + 48);
      if ( v53 )
      {
LABEL_63:
        v62 = v33;
        sub_B91220((__int64)v33, v53);
        v33 = v62;
      }
      v54 = (unsigned __int8 *)v67[0];
      *(const char **)(v36 + 48) = v67[0];
      if ( v54 )
        sub_B976B0((__int64)v67, v54, (__int64)v33);
      goto LABEL_37;
    }
  }
  return v18;
}
