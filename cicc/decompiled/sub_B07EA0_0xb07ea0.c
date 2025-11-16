// Function: sub_B07EA0
// Address: 0xb07ea0
//
__int64 __fastcall sub_B07EA0(
        __int64 *a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __int64 a7,
        int a8,
        __int64 a9,
        int a10,
        int a11,
        int a12,
        unsigned int a13,
        __int64 a14,
        __int64 a15,
        unsigned __int64 a16,
        unsigned __int64 a17,
        __int64 a18,
        __int64 a19,
        __int64 a20,
        unsigned int a21,
        char a22)
{
  __int64 v22; // r10
  __int64 v23; // r11
  int v24; // r13d
  __int64 v25; // r12
  __int64 v26; // r15
  int v27; // ebx
  char v28; // dl
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rdx
  int v32; // eax
  __int64 v33; // r10
  __int64 v34; // r11
  __int64 v35; // r15
  __int64 v36; // r12
  unsigned int v37; // eax
  __int64 v38; // rdx
  __int64 v39; // rbx
  __int64 result; // rax
  __int64 v41; // rsi
  __int64 v42; // r12
  __int64 v43; // rax
  __int64 v44; // rdi
  __int64 v45; // [rsp+8h] [rbp-148h]
  __int64 v46; // [rsp+18h] [rbp-138h]
  __int64 v47; // [rsp+20h] [rbp-130h]
  __int64 v48; // [rsp+30h] [rbp-120h]
  __int64 v49; // [rsp+38h] [rbp-118h]
  int v51; // [rsp+48h] [rbp-108h]
  __int64 v52; // [rsp+50h] [rbp-100h]
  __int64 v53; // [rsp+50h] [rbp-100h]
  __int64 v54; // [rsp+50h] [rbp-100h]
  __int64 v55; // [rsp+58h] [rbp-F8h]
  __int64 v56; // [rsp+60h] [rbp-F0h]
  unsigned int v57; // [rsp+60h] [rbp-F0h]
  __int64 v58; // [rsp+60h] [rbp-F0h]
  __int64 v61; // [rsp+70h] [rbp-E0h]
  __int64 v62; // [rsp+70h] [rbp-E0h]
  __int64 v63; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v64; // [rsp+88h] [rbp-C8h]
  __int64 *v65; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v66; // [rsp+98h] [rbp-B8h] BYREF
  __int64 v67; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v68; // [rsp+A8h] [rbp-A8h] BYREF
  __int64 v69; // [rsp+B0h] [rbp-A0h] BYREF
  __int64 v70; // [rsp+B8h] [rbp-98h] BYREF
  __int64 v71; // [rsp+C0h] [rbp-90h]
  __int64 v72; // [rsp+C8h] [rbp-88h]
  unsigned __int64 v73; // [rsp+D0h] [rbp-80h]
  unsigned __int64 v74; // [rsp+D8h] [rbp-78h]
  __int64 v75; // [rsp+E0h] [rbp-70h]
  __int64 v76; // [rsp+E8h] [rbp-68h]
  __int64 v77; // [rsp+F0h] [rbp-60h]
  __int64 v78; // [rsp+F8h] [rbp-58h]
  __int64 v79; // [rsp+100h] [rbp-50h]
  __int64 v80; // [rsp+108h] [rbp-48h]
  __int64 v81; // [rsp+110h] [rbp-40h]

  v22 = a3;
  v23 = a4;
  v24 = (int)a1;
  v25 = (__int64)a2;
  if ( a21 )
  {
LABEL_33:
    v68 = v25;
    v65 = &v67;
    v41 = 13;
    v67 = a5;
    v69 = v22;
    v71 = a7;
    v70 = v23;
    v72 = a14;
    v73 = a16;
    v74 = a17;
    v75 = a9;
    v76 = a15;
    v77 = a18;
    v78 = a19;
    v79 = a20;
    v66 = 0xD0000000DLL;
    if ( !a20 )
    {
      if ( a19 )
      {
        LODWORD(v66) = 12;
        v41 = 12;
      }
      else if ( a18 )
      {
        LODWORD(v66) = 11;
        v41 = 11;
      }
      else if ( a15 )
      {
        LODWORD(v66) = 10;
        v41 = 10;
      }
      else if ( a9 )
      {
        LODWORD(v66) = 9;
        v41 = 9;
      }
      else
      {
        LODWORD(v66) = 8;
        v41 = 8;
      }
    }
    v42 = *a1 + 1048;
    v43 = sub_B97910(40, v41, a21);
    v44 = v43;
    if ( v43 )
    {
      v61 = v43;
      sub_AF3420(v43, v24, a21, a6, a8, a10, a11, a12, a13, (__int64)&v67, v41);
      v44 = v61;
    }
    result = sub_B07DC0(v44, a21, v42);
    if ( v65 != &v67 )
    {
      v62 = result;
      _libc_free(v65, a21);
      return v62;
    }
    return result;
  }
  v26 = *a1;
  v65 = a2;
  v66 = a3;
  v68 = a5;
  v67 = a4;
  LODWORD(v69) = a6;
  v70 = a7;
  LODWORD(v71) = a8;
  v72 = a9;
  v73 = __PAIR64__(a11, a10);
  v74 = __PAIR64__(a13, a12);
  v75 = a14;
  v76 = a15;
  v77 = a16;
  v78 = a17;
  v79 = a18;
  v80 = a19;
  v81 = a20;
  v27 = *(_DWORD *)(v26 + 1072);
  v55 = *(_QWORD *)(v26 + 1056);
  if ( !v27 )
    goto LABEL_32;
  v63 = 0;
  v28 = a13;
  v64 = 0;
  if ( a2 )
  {
    if ( *(_BYTE *)a2 == 14 )
    {
      v52 = v22;
      v29 = sub_AF5140((__int64)a2, 7u);
      v28 = a13;
      v22 = v52;
      v23 = a4;
      if ( v29 )
      {
        v30 = sub_B91420(v29, 7);
        v22 = v52;
        v23 = a4;
        v63 = v30;
        v64 = v31;
        v28 = BYTE4(v74);
      }
    }
  }
  if ( (v28 & 8) == 0 && v67 && v65 && *(_BYTE *)v65 == 14 )
  {
    v54 = v23;
    v58 = v22;
    v32 = sub_AFA7A0(&v67, &v63);
    v34 = v54;
    v33 = v58;
  }
  else
  {
    v53 = v23;
    v56 = v22;
    v32 = sub_AFA420(&v66, &v63, &v68, &v70, (int *)&v69);
    v33 = v56;
    v34 = v53;
  }
  v57 = (v27 - 1) & v32;
  v51 = 1;
  v49 = v33;
  v48 = v26;
  v35 = v34;
  while ( 1 )
  {
    v36 = *(_QWORD *)(v55 + 8LL * v57);
    if ( v36 == -4096 )
    {
      v25 = (__int64)a2;
      v22 = v49;
      v23 = v35;
      goto LABEL_32;
    }
    if ( v36 != -8192 )
    {
      if ( (v74 & 0x800000000LL) == 0 && v67 != 0 && v65 != 0 )
      {
        v47 = v67;
        if ( *(_BYTE *)v65 == 14 )
        {
          v46 = (__int64)v65;
          if ( sub_AF5140((__int64)v65, 7u) )
          {
            if ( (*(_BYTE *)(v36 + 36) & 8) == 0 )
            {
              v45 = v76;
              if ( v46 == *((_QWORD *)sub_A17150((_BYTE *)(v36 - 16)) + 1) && v47 == sub_AF5140(v36, 3u) )
              {
                if ( (*(_BYTE *)(v36 - 16) & 2) != 0 )
                  v37 = *(_DWORD *)(v36 - 24);
                else
                  v37 = (*(_WORD *)(v36 - 16) >> 6) & 0xF;
                v38 = 0;
                if ( v37 > 9 )
                  v38 = *((_QWORD *)sub_A17150((_BYTE *)(v36 - 16)) + 9);
                if ( v45 == v38 )
                  break;
              }
            }
          }
        }
      }
      if ( sub_AF5710((__int64 *)&v65, v36) )
        break;
    }
    v57 = (v27 - 1) & (v51 + v57);
    ++v51;
  }
  v23 = v35;
  v39 = v36;
  v22 = v49;
  v25 = (__int64)a2;
  if ( v55 + 8LL * v57 == *(_QWORD *)(v48 + 1056) + 8LL * *(unsigned int *)(v48 + 1072) || (result = v39) == 0 )
  {
LABEL_32:
    result = 0;
    if ( !a22 )
      return result;
    goto LABEL_33;
  }
  return result;
}
