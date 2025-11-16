// Function: sub_2D62F50
// Address: 0x2d62f50
//
__int64 __fastcall sub_2D62F50(
        unsigned __int64 a1,
        __int64 a2,
        __int64 a3,
        _DWORD *a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        unsigned __int8 a8)
{
  __int64 *v11; // rdx
  __int64 v12; // r12
  __int64 v13; // rax
  _QWORD *v14; // rax
  __int64 v15; // rdx
  unsigned int v16; // esi
  __int64 v17; // r8
  int v18; // r13d
  __int64 *v19; // r11
  unsigned int v20; // edx
  __int64 *v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // r14
  __int64 *v26; // rax
  __int64 v27; // r9
  __int64 *v28; // r13
  __int64 v29; // rax
  __int64 v30; // rdx
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rcx
  char *v33; // r8
  unsigned __int64 v34; // rsi
  int v35; // eax
  _QWORD *v36; // rdx
  __int64 v37; // r14
  __int64 v38; // rdx
  __int64 v39; // rcx
  unsigned __int8 *v40; // rdx
  int v41; // eax
  const void **v42; // rsi
  unsigned int v43; // edx
  __int64 v44; // rax
  __int64 v45; // rax
  unsigned int v46; // edx
  __int64 *v48; // rax
  __int64 *v49; // r10
  __int64 v50; // rax
  __int64 v51; // rax
  unsigned __int64 v52; // rdi
  __int64 v53; // r8
  unsigned __int64 v54; // rsi
  unsigned __int64 v55; // r9
  char *v56; // rcx
  int v57; // edx
  _QWORD *v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rcx
  unsigned __int8 *v61; // r8
  int v62; // edi
  int v63; // ecx
  __int64 v64; // rdi
  char *v65; // r14
  __int64 **v66; // [rsp+0h] [rbp-160h]
  unsigned __int64 v67; // [rsp+8h] [rbp-158h]
  __int64 *v68; // [rsp+18h] [rbp-148h]
  __int64 *v69; // [rsp+20h] [rbp-140h]
  __int64 v70; // [rsp+28h] [rbp-138h]
  __int64 *v71; // [rsp+28h] [rbp-138h]
  __int64 v73; // [rsp+38h] [rbp-128h]
  unsigned __int8 *v74; // [rsp+38h] [rbp-128h]
  char *v75; // [rsp+38h] [rbp-128h]
  __int64 v76; // [rsp+38h] [rbp-128h]
  int v79; // [rsp+50h] [rbp-110h]
  __int64 v80; // [rsp+58h] [rbp-108h]
  int v81; // [rsp+68h] [rbp-F8h]
  __int64 v82[4]; // [rsp+70h] [rbp-F0h] BYREF
  char v83; // [rsp+90h] [rbp-D0h]
  char v84; // [rsp+91h] [rbp-CFh]
  __int64 *v85; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v86; // [rsp+A8h] [rbp-B8h]
  _BYTE v87[32]; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v88; // [rsp+D0h] [rbp-90h]
  __int64 v89; // [rsp+D8h] [rbp-88h]
  __int16 v90; // [rsp+E0h] [rbp-80h]
  __int64 v91; // [rsp+E8h] [rbp-78h]
  void **v92; // [rsp+F0h] [rbp-70h]
  void **v93; // [rsp+F8h] [rbp-68h]
  __int64 v94; // [rsp+100h] [rbp-60h]
  int v95; // [rsp+108h] [rbp-58h]
  __int16 v96; // [rsp+10Ch] [rbp-54h]
  char v97; // [rsp+10Eh] [rbp-52h]
  __int64 v98; // [rsp+110h] [rbp-50h]
  __int64 v99; // [rsp+118h] [rbp-48h]
  void *v100; // [rsp+120h] [rbp-40h] BYREF
  void *v101; // [rsp+128h] [rbp-38h] BYREF

  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v11 = *(__int64 **)(a1 - 8);
  else
    v11 = (__int64 *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  v12 = *v11;
  *a4 = 0;
  v13 = *(_QWORD *)(v12 + 16);
  if ( !v13 || *(_QWORD *)(v13 + 8) )
  {
    v14 = (_QWORD *)sub_2D5E810(a2, a1, *(__int64 ***)(v12 + 8));
    v15 = (__int64)v14;
    if ( *(_BYTE *)v14 > 0x1Cu )
    {
      v80 = (__int64)v14;
      sub_B44530(v14, v12);
      v15 = v80;
      if ( a6 )
      {
        sub_9C95B0(a6, v80);
        v15 = v80;
      }
    }
    sub_2D58400(a2, v12, v15);
    sub_2D598D0(a2, a1, 0, v12);
  }
  v82[0] = v12;
  v16 = *(_DWORD *)(a3 + 24);
  if ( !v16 )
  {
    v85 = 0;
    ++*(_QWORD *)a3;
    goto LABEL_79;
  }
  v17 = *(_QWORD *)(a3 + 8);
  v18 = 1;
  v19 = 0;
  v20 = (v16 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
  v21 = (__int64 *)(v17 + 16LL * v20);
  v22 = *v21;
  if ( v12 != *v21 )
  {
    while ( v22 != -4096 )
    {
      if ( v22 == -8192 && !v19 )
        v19 = v21;
      v20 = (v16 - 1) & (v18 + v20);
      v21 = (__int64 *)(v17 + 16LL * v20);
      v22 = *v21;
      if ( v12 == *v21 )
        goto LABEL_10;
      ++v18;
    }
    if ( v19 )
      v21 = v19;
    ++*(_QWORD *)a3;
    v62 = *(_DWORD *)(a3 + 16);
    v85 = v21;
    v63 = v62 + 1;
    if ( 4 * (v62 + 1) < 3 * v16 )
    {
      v64 = v12;
      if ( v16 - *(_DWORD *)(a3 + 20) - v63 <= v16 >> 3 )
      {
        sub_2D58F00(a3, v16);
        sub_2D56820(a3, v82, &v85);
        v64 = v82[0];
        v63 = *(_DWORD *)(a3 + 16) + 1;
        v21 = v85;
      }
      goto LABEL_70;
    }
LABEL_79:
    sub_2D58F00(a3, 2 * v16);
    sub_2D56820(a3, v82, &v85);
    v64 = v82[0];
    v63 = *(_DWORD *)(a3 + 16) + 1;
    v21 = v85;
LABEL_70:
    *(_DWORD *)(a3 + 16) = v63;
    if ( *v21 != -4096 )
      --*(_DWORD *)(a3 + 20);
    *v21 = v64;
    v21[1] = 0;
    v24 = 2LL * a8;
    v23 = v82[0];
    goto LABEL_12;
  }
LABEL_10:
  if ( a8 == ((v21[1] >> 1) & 3) )
    goto LABEL_13;
  v23 = v12;
  v24 = 4;
LABEL_12:
  v21[1] = v24 | *(_QWORD *)(v23 + 8) & 0xFFFFFFFFFFFFFFF9LL;
LABEL_13:
  v25 = *(_QWORD *)(a1 + 8);
  v26 = (__int64 *)sub_22077B0(0x18u);
  v28 = v26;
  if ( v26 )
  {
    v26[1] = v12;
    *v26 = (__int64)off_49D4120;
    v29 = *(_QWORD *)(v12 + 8);
    *(_QWORD *)(v12 + 8) = v25;
    v28[2] = v29;
  }
  v30 = *(unsigned int *)(a2 + 8);
  v31 = *(unsigned int *)(a2 + 12);
  v85 = v28;
  v32 = *(_QWORD *)a2;
  v33 = (char *)&v85;
  v34 = v30 + 1;
  v35 = v30;
  if ( v30 + 1 > v31 )
  {
    if ( v32 > (unsigned __int64)&v85 || (unsigned __int64)&v85 >= v32 + 8 * v30 )
    {
      sub_2D57B00(a2, v34, v30, v32, (__int64)&v85, v27);
      v30 = *(unsigned int *)(a2 + 8);
      v32 = *(_QWORD *)a2;
      v33 = (char *)&v85;
      v35 = *(_DWORD *)(a2 + 8);
    }
    else
    {
      v65 = (char *)&v85 - v32;
      sub_2D57B00(a2, v34, v30, v32, (__int64)&v85, v27);
      v32 = *(_QWORD *)a2;
      v30 = *(unsigned int *)(a2 + 8);
      v33 = &v65[*(_QWORD *)a2];
      v35 = *(_DWORD *)(a2 + 8);
    }
  }
  v36 = (_QWORD *)(v32 + 8 * v30);
  if ( v36 )
  {
    *v36 = *(_QWORD *)v33;
    *(_QWORD *)v33 = 0;
    v28 = v85;
    v35 = *(_DWORD *)(a2 + 8);
  }
  *(_DWORD *)(a2 + 8) = v35 + 1;
  if ( v28 )
    (*(void (__fastcall **)(__int64 *, unsigned __int64, _QWORD *, unsigned __int64, char *))(*v28 + 8))(
      v28,
      v34,
      v36,
      v32,
      v33);
  v37 = 0;
  sub_2D58400(a2, a1, v12);
  v79 = *(_DWORD *)(v12 + 4) & 0x7FFFFFF;
  if ( v79 )
  {
    while ( 1 )
    {
      if ( (*(_BYTE *)(v12 + 7) & 0x40) != 0 )
        v38 = *(_QWORD *)(v12 - 8);
      else
        v38 = v12 - 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF);
      v39 = *(_QWORD *)(a1 + 8);
      v40 = *(unsigned __int8 **)(v38 + 32 * v37);
      if ( *((_QWORD *)v40 + 1) == v39 || *(_BYTE *)v12 == 86 && !(_DWORD)v37 )
        goto LABEL_32;
      v41 = *v40;
      if ( (_BYTE)v41 == 17 )
      {
        v42 = (const void **)(v40 + 24);
        v43 = *(_DWORD *)(v39 + 8) >> 8;
        if ( a8 )
          sub_C44830((__int64)&v85, v42, v43);
        else
          sub_C449B0((__int64)&v85, v42, v43);
        v44 = sub_AD8D80(*(_QWORD *)(a1 + 8), (__int64)&v85);
        sub_2D598D0(a2, v12, v37, v44);
        if ( (unsigned int)v86 > 0x40 && v85 )
          j_j___libc_free_0_0((unsigned __int64)v85);
LABEL_32:
        if ( v79 == (_DWORD)++v37 )
          break;
      }
      else
      {
        if ( (unsigned int)(v41 - 12) > 1 )
        {
          if ( a8 )
          {
            v66 = *(__int64 ***)(a1 + 8);
            v67 = (unsigned __int64)v40;
            v48 = (__int64 *)sub_22077B0(0x18u);
            v49 = v48;
            if ( v48 )
            {
              v48[1] = v12;
              *v48 = (__int64)off_49D40C0;
              v68 = v48;
              v50 = sub_BD5C60(v12);
              v97 = 7;
              v91 = v50;
              v93 = &v101;
              v96 = 512;
              v86 = 0x200000000LL;
              v85 = (__int64 *)v87;
              v92 = &v100;
              v90 = 0;
              v100 = &unk_49DA100;
              v101 = &unk_49DA0B0;
              v94 = 0;
              v95 = 0;
              v98 = 0;
              v99 = 0;
              v88 = 0;
              v89 = 0;
              sub_D5F1F0((__int64)&v85, v12);
              v82[0] = (__int64)"promoted";
              v84 = 1;
              v83 = 3;
              v68[2] = sub_2D5B7B0((__int64 *)&v85, 0x28u, v67, v66, (__int64)v82, 0, v81, 0);
              nullsub_61();
              v100 = &unk_49DA100;
              nullsub_63();
              v49 = v68;
              if ( v85 != (__int64 *)v87 )
              {
                _libc_free((unsigned __int64)v85);
                v49 = v68;
              }
            }
            v51 = *(unsigned int *)(a2 + 8);
            v52 = *(unsigned int *)(a2 + 12);
            v85 = v49;
            v53 = v49[2];
            v54 = *(_QWORD *)a2;
            v55 = v51 + 1;
            v56 = (char *)&v85;
            v57 = v51;
            if ( v51 + 1 > v52 )
            {
              if ( v54 > (unsigned __int64)&v85 || (unsigned __int64)&v85 >= v54 + 8 * v51 )
              {
                v71 = v49;
                v76 = v49[2];
                sub_2D57B00(a2, v55, v51, (__int64)&v85, v53, v55);
                v51 = *(unsigned int *)(a2 + 8);
                v54 = *(_QWORD *)a2;
                v56 = (char *)&v85;
                v49 = v71;
                v53 = v76;
                v57 = *(_DWORD *)(a2 + 8);
              }
              else
              {
                v69 = v49;
                v75 = (char *)&v85 - v54;
                v70 = v49[2];
                sub_2D57B00(a2, v55, v51, (__int64)&v85, v53, v55);
                v54 = *(_QWORD *)a2;
                v51 = *(unsigned int *)(a2 + 8);
                v53 = v70;
                v49 = v69;
                v57 = *(_DWORD *)(a2 + 8);
                v56 = &v75[*(_QWORD *)a2];
              }
            }
            v58 = (_QWORD *)(v54 + 8 * v51);
            if ( v58 )
            {
              *v58 = *(_QWORD *)v56;
              *(_QWORD *)v56 = 0;
              v49 = v85;
              ++*(_DWORD *)(a2 + 8);
              if ( v49 )
                goto LABEL_46;
            }
            else
            {
              *(_DWORD *)(a2 + 8) = v57 + 1;
LABEL_46:
              v73 = v53;
              (*(void (__fastcall **)(__int64 *))(*v49 + 8))(v49);
              v53 = v73;
            }
          }
          else
          {
            v53 = sub_2D5EBE0(a2, v12, (unsigned __int64)v40, (__int64 **)v39);
          }
          v74 = (unsigned __int8 *)v53;
          sub_2D598D0(a2, v12, v37, v53);
          v61 = v74;
          if ( *v74 > 0x1Cu )
          {
            if ( a5 )
            {
              sub_9C95B0(a5, (__int64)v74);
              v61 = v74;
            }
            *a4 += (unsigned __int8)sub_2D5C100(a7, v61, v59, v60, (__int64)v61) ^ 1;
          }
          goto LABEL_32;
        }
        v45 = sub_ACA8A0(*(__int64 ***)(a1 + 8));
        v46 = v37++;
        sub_2D598D0(a2, v12, v46, v45);
        if ( v79 == (_DWORD)v37 )
          break;
      }
    }
  }
  sub_2D5CED0(a2, a1, 0);
  return v12;
}
