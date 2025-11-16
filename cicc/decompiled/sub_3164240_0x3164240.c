// Function: sub_3164240
// Address: 0x3164240
//
__int64 __fastcall sub_3164240(__int64 a1, __int64 a2, char a3, __int64 a4, unsigned __int8 *a5, _QWORD *a6, char a7)
{
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  __int16 v15; // dx
  __int16 v16; // cx
  __int64 v17; // rdx
  char v18; // si
  char v19; // al
  __int16 v20; // cx
  char v21; // al
  char v22; // di
  char i; // si
  unsigned __int64 v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  unsigned __int8 *v28; // r14
  unsigned __int8 v29; // al
  __int64 v30; // r15
  __int64 v31; // rbx
  __int64 v32; // r8
  __int64 v33; // rax
  char *v35; // rcx
  unsigned int v36; // eax
  int v37; // eax
  unsigned int v38; // esi
  unsigned int v39; // edx
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rax
  _QWORD *v43; // rax
  __int64 v44; // rbx
  _BYTE *v45; // r13
  __int64 v46; // r12
  __int64 v47; // rdx
  unsigned int v48; // esi
  __int64 v49; // [rsp+8h] [rbp-1C8h]
  unsigned __int8 v50; // [rsp+10h] [rbp-1C0h]
  _QWORD *v51; // [rsp+10h] [rbp-1C0h]
  __int64 *v53; // [rsp+18h] [rbp-1B8h]
  __int64 v54; // [rsp+18h] [rbp-1B8h]
  __int64 *v56; // [rsp+20h] [rbp-1B0h]
  __int64 v57; // [rsp+48h] [rbp-188h] BYREF
  const char *v58; // [rsp+50h] [rbp-180h] BYREF
  __int64 v59; // [rsp+58h] [rbp-178h]
  _QWORD v60[2]; // [rsp+60h] [rbp-170h] BYREF
  __int16 v61; // [rsp+70h] [rbp-160h]
  _BYTE *v62; // [rsp+80h] [rbp-150h] BYREF
  __int64 v63; // [rsp+88h] [rbp-148h]
  _BYTE v64[32]; // [rsp+90h] [rbp-140h] BYREF
  __int64 v65; // [rsp+B0h] [rbp-120h]
  __int64 v66; // [rsp+B8h] [rbp-118h]
  __int64 v67; // [rsp+C0h] [rbp-110h]
  __int64 v68; // [rsp+C8h] [rbp-108h]
  void **v69; // [rsp+D0h] [rbp-100h]
  void **v70; // [rsp+D8h] [rbp-F8h]
  __int64 v71; // [rsp+E0h] [rbp-F0h]
  int v72; // [rsp+E8h] [rbp-E8h]
  __int16 v73; // [rsp+ECh] [rbp-E4h]
  char v74; // [rsp+EEh] [rbp-E2h]
  __int64 v75; // [rsp+F0h] [rbp-E0h]
  __int64 v76; // [rsp+F8h] [rbp-D8h]
  void *v77; // [rsp+100h] [rbp-D0h] BYREF
  void *v78; // [rsp+108h] [rbp-C8h] BYREF
  char *v79; // [rsp+110h] [rbp-C0h] BYREF
  __int64 v80; // [rsp+118h] [rbp-B8h]
  _BYTE v81[16]; // [rsp+120h] [rbp-B0h] BYREF
  __int16 v82; // [rsp+130h] [rbp-A0h]

  v12 = sub_B2BE50(a4);
  v13 = *(_QWORD *)(a4 + 80);
  v68 = v12;
  v69 = &v77;
  v70 = &v78;
  v62 = v64;
  v77 = &unk_49DA100;
  v63 = 0x200000000LL;
  v71 = 0;
  v78 = &unk_49DA0B0;
  if ( v13 )
    v13 -= 24;
  v73 = 512;
  v72 = 0;
  v74 = 7;
  v75 = 0;
  v76 = 0;
  v65 = 0;
  v66 = 0;
  LOWORD(v67) = 0;
  v14 = sub_AA5190(v13);
  v16 = v15;
  v17 = v14;
  if ( v14 )
  {
    v18 = v16;
    v19 = HIBYTE(v16);
  }
  else
  {
    v19 = 0;
    v18 = 0;
  }
  LOBYTE(v20) = v18;
  HIBYTE(v20) = v19;
  v21 = 0;
  v22 = v18;
  for ( i = HIBYTE(v20); ; i = 0 )
  {
    if ( !v17 )
      BUG();
    if ( *(_BYTE *)(v17 - 24) != 85 )
      break;
    v32 = *(_QWORD *)(v17 - 56);
    if ( !v32 || *(_BYTE *)v32 || *(_QWORD *)(v32 + 24) != *(_QWORD *)(v17 + 56) || (*(_BYTE *)(v32 + 33) & 0x20) == 0 )
      break;
    v17 = *(_QWORD *)(v17 + 8);
    v21 = 1;
    v22 = 0;
  }
  if ( v21 )
  {
    LOBYTE(v20) = v22;
    HIBYTE(v20) = i;
  }
  v24 = *(_QWORD *)(a4 + 80);
  if ( v24 )
    v24 -= 24LL;
  sub_A88F30((__int64)&v62, v24, v17, v20);
  if ( !a5 )
  {
LABEL_34:
    *(_BYTE *)(a1 + 16) = 0;
    goto LABEL_37;
  }
  v28 = a5;
  v29 = *a5;
  if ( *a5 <= 0x1Cu )
  {
LABEL_20:
    v30 = (__int64)v28;
    goto LABEL_21;
  }
  while ( 1 )
  {
    if ( v29 == 61 )
    {
      v28 = (unsigned __int8 *)*((_QWORD *)v28 - 4);
      if ( !a7 )
      {
        v24 = 1;
        a6 = (_QWORD *)sub_B0DAC0(a6, 1, 0);
      }
      if ( !v28 )
        goto LABEL_34;
      goto LABEL_18;
    }
    if ( v29 != 62 )
      break;
    v28 = (unsigned __int8 *)*((_QWORD *)v28 - 8);
    if ( !v28 )
      goto LABEL_34;
LABEL_18:
    a7 = 0;
LABEL_19:
    v29 = *v28;
    if ( *v28 <= 0x1Cu )
      goto LABEL_20;
  }
  v24 = 0;
  v59 = 0;
  v80 = 0x1000000000LL;
  v79 = v81;
  v58 = (const char *)v60;
  if ( a6 )
    v24 = sub_AF4EB0((__int64)a6);
  v49 = sub_F53E50(v28, v24, (__int64)&v79, (__int64)&v58);
  if ( v49 && !(_DWORD)v59 )
  {
    v24 = (unsigned __int64)v79;
    a6 = (_QWORD *)sub_B0DBA0(a6, v79, (unsigned int)v80, 0, 0);
    if ( v58 != (const char *)v60 )
      _libc_free((unsigned __int64)v58);
    if ( v79 != v81 )
      _libc_free((unsigned __int64)v79);
    v28 = (unsigned __int8 *)v49;
    a7 = 0;
    goto LABEL_19;
  }
  v30 = (__int64)v28;
  if ( v58 != (const char *)v60 )
    _libc_free((unsigned __int64)v58);
  if ( v79 != v81 )
    _libc_free((unsigned __int64)v79);
  v29 = *v28;
LABEL_21:
  if ( v29 != 22 )
  {
    v57 = 0;
    goto LABEL_36;
  }
  v24 = 73;
  v57 = v30;
  if ( (unsigned __int8)sub_B2D670(v30, 73) )
  {
    if ( a3 && !sub_AF46F0((__int64)a6) && sub_AF4590((__int64)a6) )
    {
      v24 = 8;
      a6 = (_QWORD *)sub_B0DAC0(a6, 8, 0);
    }
    goto LABEL_36;
  }
  if ( (unsigned __int8)sub_3163180(a2, &v57, &v58) )
  {
    v56 = (__int64 *)(v58 + 8);
    goto LABEL_25;
  }
  v35 = (char *)v58;
  v36 = *(_DWORD *)(a2 + 8);
  ++*(_QWORD *)a2;
  v79 = v35;
  v37 = (v36 >> 1) + 1;
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v39 = 12;
    v38 = 4;
  }
  else
  {
    v38 = *(_DWORD *)(a2 + 24);
    v39 = 3 * v38;
  }
  if ( 4 * v37 >= v39 )
  {
    v38 *= 2;
LABEL_75:
    sub_3163E20(a2, v38);
    sub_3163180(a2, &v57, &v79);
    v35 = v79;
    v37 = (*(_DWORD *)(a2 + 8) >> 1) + 1;
    goto LABEL_58;
  }
  if ( v38 - (v37 + *(_DWORD *)(a2 + 12)) <= v38 >> 3 )
    goto LABEL_75;
LABEL_58:
  *(_DWORD *)(a2 + 8) = *(_DWORD *)(a2 + 8) & 1 | (2 * v37);
  if ( *(_QWORD *)v35 != -4096 )
    --*(_DWORD *)(a2 + 12);
  v40 = v57;
  *((_QWORD *)v35 + 1) = 0;
  *(_QWORD *)v35 = v40;
  v56 = (__int64 *)(v35 + 8);
LABEL_25:
  v31 = *v56;
  if ( !*v56 )
  {
    v58 = sub_BD5D20(v30);
    v61 = 773;
    v59 = v41;
    v60[0] = ".debug";
    v53 = *(__int64 **)(v30 + 8);
    v42 = sub_AA4E30(v65);
    v50 = sub_AE5260(v42, (__int64)v53);
    v82 = 257;
    v43 = sub_BD2C40(80, 1u);
    v44 = (__int64)v43;
    if ( v43 )
      sub_B4CCA0((__int64)v43, v53, 0, 0, v50, (__int64)&v79, 0, 0);
    (*((void (__fastcall **)(void **, __int64, const char **, __int64, __int64))*v70 + 2))(v70, v44, &v58, v66, v67);
    if ( v62 != &v62[16 * (unsigned int)v63] )
    {
      v54 = a1;
      v45 = &v62[16 * (unsigned int)v63];
      v51 = a6;
      v46 = (__int64)v62;
      do
      {
        v47 = *(_QWORD *)(v46 + 8);
        v48 = *(_DWORD *)v46;
        v46 += 16;
        sub_B99FD0(v44, v48, v47);
      }
      while ( v45 != (_BYTE *)v46 );
      a1 = v54;
      a6 = v51;
    }
    *v56 = v44;
    sub_315E620((__int64 *)&v62, v30, v44, 0, 0);
    v31 = *v56;
  }
  v24 = 1;
  v30 = v31;
  a6 = (_QWORD *)sub_B0DAC0(a6, 1, 0);
LABEL_36:
  v33 = sub_E3D320(a6, v24, v25, v26, v27);
  *(_QWORD *)a1 = v30;
  *(_QWORD *)(a1 + 8) = v33;
  *(_BYTE *)(a1 + 16) = 1;
LABEL_37:
  nullsub_61();
  v77 = &unk_49DA100;
  nullsub_63();
  if ( v62 != v64 )
    _libc_free((unsigned __int64)v62);
  return a1;
}
