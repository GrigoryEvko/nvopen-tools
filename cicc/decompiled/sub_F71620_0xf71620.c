// Function: sub_F71620
// Address: 0xf71620
//
_QWORD *__fastcall sub_F71620(_QWORD *a1, __int64 a2, __int64 *a3, __int64 a4, __int64 *a5, char a6)
{
  __int64 *v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r11
  _QWORD *v15; // r10
  __int64 v16; // r14
  __int64 v17; // r15
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v21; // rdx
  _QWORD *v22; // rax
  __int64 v23; // rbx
  unsigned int *v24; // r13
  unsigned int *v25; // r15
  __int64 v26; // rdx
  unsigned int v27; // esi
  __int64 v28; // rdx
  _QWORD *v29; // rax
  __int64 v30; // r13
  __int64 v31; // rsi
  unsigned int *v32; // r15
  unsigned int *v33; // r14
  __int64 v34; // rdx
  __int64 *v35; // r15
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // rax
  __int64 v40; // rax
  bool v41; // al
  __int64 v42; // rax
  __int64 v43; // rdi
  bool v44; // al
  __int64 v45; // rdx
  __int64 v46; // rax
  char v47; // al
  bool v48; // zf
  __int64 v49; // rax
  _QWORD *v50; // [rsp+10h] [rbp-150h]
  __int64 v51; // [rsp+10h] [rbp-150h]
  __int64 v52; // [rsp+18h] [rbp-148h]
  _QWORD *v53; // [rsp+18h] [rbp-148h]
  __int64 v54; // [rsp+18h] [rbp-148h]
  _QWORD *v55; // [rsp+20h] [rbp-140h]
  _QWORD *v56; // [rsp+20h] [rbp-140h]
  __int64 v57; // [rsp+20h] [rbp-140h]
  _QWORD *v58; // [rsp+20h] [rbp-140h]
  __int64 v60; // [rsp+28h] [rbp-138h]
  __int64 v61; // [rsp+28h] [rbp-138h]
  __int64 v63; // [rsp+38h] [rbp-128h]
  const char *v64; // [rsp+40h] [rbp-120h] BYREF
  __int64 v65; // [rsp+48h] [rbp-118h]
  char *v66; // [rsp+50h] [rbp-110h]
  __int16 v67; // [rsp+60h] [rbp-100h]
  _BYTE v68[32]; // [rsp+70h] [rbp-F0h] BYREF
  __int16 v69; // [rsp+90h] [rbp-D0h]
  unsigned int *v70; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v71; // [rsp+A8h] [rbp-B8h]
  _BYTE v72[32]; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v73; // [rsp+D0h] [rbp-90h]
  __int64 v74; // [rsp+D8h] [rbp-88h]
  __int64 v75; // [rsp+E0h] [rbp-80h]
  __int64 v76; // [rsp+E8h] [rbp-78h]
  void **v77; // [rsp+F0h] [rbp-70h]
  void **v78; // [rsp+F8h] [rbp-68h]
  __int64 v79; // [rsp+100h] [rbp-60h]
  int v80; // [rsp+108h] [rbp-58h]
  __int16 v81; // [rsp+10Ch] [rbp-54h]
  char v82; // [rsp+10Eh] [rbp-52h]
  __int64 v83; // [rsp+110h] [rbp-50h]
  __int64 v84; // [rsp+118h] [rbp-48h]
  void *v85; // [rsp+120h] [rbp-40h] BYREF
  void *v86; // [rsp+128h] [rbp-38h] BYREF

  v9 = (__int64 *)sub_BD5C60(a4);
  v10 = sub_BCE3C0(v9, *(_DWORD *)(a2 + 40));
  v14 = *(_QWORD *)(a2 + 8);
  v15 = *(_QWORD **)a2;
  v63 = 0;
  v16 = v10;
  if ( a6 )
  {
    v63 = *a3;
    if ( *a3 )
    {
      if ( *((_WORD *)v15 + 12) == 8
        && *(_WORD *)(v14 + 24) == 8
        && (v52 = *(_QWORD *)(a2 + 8),
            v56 = *(_QWORD **)a2,
            v35 = (__int64 *)*a5,
            v61 = sub_D33D80((_QWORD *)v14, *a5, v11, v12, v13),
            v39 = sub_D33D80(v56, (__int64)v35, v36, v37, v38),
            v15 = v56,
            v14 = v52,
            v61 == v39)
        && v56[6] == v63
        && *(_QWORD *)(v52 + 48) == v63 )
      {
        v50 = v56;
        v40 = sub_D47930(v63);
        v57 = sub_DBA6E0((__int64)v35, v63, v40, 0);
        v41 = sub_D96A50(v57);
        v63 = 0;
        v14 = v52;
        v15 = v50;
        if ( !v41 )
        {
          v42 = sub_D95540(v57);
          v14 = v52;
          v15 = v50;
          if ( *(_BYTE *)(v42 + 8) == 12 )
          {
            v43 = (__int64)v50;
            v51 = v52;
            v53 = v15;
            v58 = sub_DD0540(v43, v57, v35);
            v44 = sub_D96A50((__int64)v58);
            v15 = v53;
            v14 = v51;
            if ( !v44 )
            {
              v45 = v53[6];
              v54 = **(_QWORD **)(v51 + 32);
              v46 = sub_DE4F70(v35, v61, v45);
              v47 = sub_DBED40((__int64)v35, v46);
              v15 = v58;
              v14 = v54;
              v48 = v47 == 0;
              v49 = 0;
              if ( v48 )
                v49 = v61;
              v63 = v49;
            }
          }
        }
      }
      else
      {
        v63 = 0;
      }
    }
  }
  v55 = v15;
  v60 = a4 + 24;
  v17 = sub_F8DB90(a5, v14, v16, a4 + 24, 0);
  v18 = sub_F8DB90(a5, v55, v16, a4 + 24, 0);
  if ( *(_BYTE *)(a2 + 44) )
  {
    v82 = 7;
    v76 = sub_BD5C60(a4);
    v77 = &v85;
    v78 = &v86;
    v70 = (unsigned int *)v72;
    v85 = &unk_49DA100;
    v71 = 0x200000000LL;
    v81 = 512;
    LOWORD(v75) = 0;
    v86 = &unk_49DA0B0;
    v79 = 0;
    v80 = 0;
    v83 = 0;
    v84 = 0;
    v73 = 0;
    v74 = 0;
    sub_D5F1F0((__int64)&v70, a4);
    v64 = sub_BD5D20(v17);
    v66 = ".fr";
    v67 = 773;
    v65 = v21;
    v69 = 257;
    v22 = sub_BD2C40(72, unk_3F10A14);
    v23 = (__int64)v22;
    if ( v22 )
      sub_B549F0((__int64)v22, v17, (__int64)v68, 0, 0);
    (*((void (__fastcall **)(void **, __int64, const char **, __int64, __int64))*v78 + 2))(v78, v23, &v64, v74, v75);
    v24 = &v70[4 * (unsigned int)v71];
    if ( v70 != v24 )
    {
      v25 = v70;
      do
      {
        v26 = *((_QWORD *)v25 + 1);
        v27 = *v25;
        v25 += 4;
        sub_B99FD0(v23, v27, v26);
      }
      while ( v24 != v25 );
    }
    v64 = sub_BD5D20(v18);
    v66 = ".fr";
    v69 = 257;
    v67 = 773;
    v65 = v28;
    v29 = sub_BD2C40(72, unk_3F10A14);
    v30 = (__int64)v29;
    if ( v29 )
      sub_B549F0((__int64)v29, v18, (__int64)v68, 0, 0);
    v31 = v30;
    (*((void (__fastcall **)(void **, __int64, const char **, __int64, __int64))*v78 + 2))(v78, v30, &v64, v74, v75);
    v32 = v70;
    v33 = &v70[4 * (unsigned int)v71];
    if ( v70 != v33 )
    {
      do
      {
        v34 = *((_QWORD *)v32 + 1);
        v31 = *v32;
        v32 += 4;
        sub_B99FD0(v30, v31, v34);
      }
      while ( v33 != v32 );
    }
    nullsub_61();
    v85 = &unk_49DA100;
    nullsub_63();
    if ( v70 != (unsigned int *)v72 )
      _libc_free(v70, v31);
    v18 = v30;
    v17 = v23;
  }
  if ( v63 )
  {
    v19 = sub_D95540(v63);
    v63 = sub_F8DB90(a5, v63, v19, v60, 0);
  }
  *a1 = 6;
  a1[1] = 0;
  if ( v17 )
  {
    a1[2] = v17;
    if ( v17 != -8192 && v17 != -4096 )
      sub_BD73F0((__int64)a1);
  }
  else
  {
    a1[2] = 0;
  }
  a1[3] = 6;
  a1[4] = 0;
  if ( v18 )
  {
    a1[5] = v18;
    if ( v18 != -4096 && v18 != -8192 )
      sub_BD73F0((__int64)(a1 + 3));
  }
  else
  {
    a1[5] = 0;
  }
  a1[6] = v63;
  return a1;
}
