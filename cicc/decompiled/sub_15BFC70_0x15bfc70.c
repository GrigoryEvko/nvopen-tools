// Function: sub_15BFC70
// Address: 0x15bfc70
//
__int64 __fastcall sub_15BFC70(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __int64 a7,
        char a8,
        char a9,
        int a10,
        __int64 a11,
        unsigned int a12,
        unsigned int a13,
        unsigned int a14,
        unsigned int a15,
        char a16,
        __int64 a17,
        __int64 a18,
        unsigned __int64 a19,
        unsigned __int64 a20,
        __int64 a21,
        unsigned int a22,
        char a23)
{
  __int64 v23; // r10
  __int64 v24; // r11
  __int64 v25; // r13
  _QWORD *v26; // r12
  __int64 v27; // rbx
  unsigned int v28; // r14d
  __int64 v29; // r15
  __int64 result; // rax
  int v31; // r9d
  __int64 v32; // rsi
  __int64 v33; // r13
  __int64 v34; // rax
  __int64 v35; // rbx
  unsigned int v36; // eax
  int v37; // r9d
  unsigned int v38; // edx
  unsigned int v39; // eax
  int v40; // r9d
  __int64 v41; // rax
  __int64 v42; // rsi
  __int64 v43; // r15
  unsigned int v44; // r12d
  __int64 *v45; // r13
  int v46; // r14d
  char v47; // bl
  __int64 v48; // rax
  __int64 v49; // rdi
  __int64 v50; // rdi
  __int64 *v51; // rdx
  char v52; // al
  __int64 v53; // [rsp+8h] [rbp-128h]
  __int64 v54; // [rsp+10h] [rbp-120h]
  __int64 v55; // [rsp+18h] [rbp-118h]
  __int64 v56; // [rsp+20h] [rbp-110h]
  int v57; // [rsp+2Ch] [rbp-104h]
  __int64 v58; // [rsp+30h] [rbp-100h]
  __int64 v59; // [rsp+38h] [rbp-F8h]
  __int64 v65; // [rsp+50h] [rbp-E0h]
  __int64 v66; // [rsp+58h] [rbp-D8h]
  int v67; // [rsp+60h] [rbp-D0h]
  int v68; // [rsp+68h] [rbp-C8h]
  int v69; // [rsp+68h] [rbp-C8h]
  unsigned int v70; // [rsp+68h] [rbp-C8h]
  __int64 v72; // [rsp+78h] [rbp-B8h]
  __int64 *v73; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v74; // [rsp+88h] [rbp-A8h] BYREF
  __int64 v75; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v76; // [rsp+98h] [rbp-98h] BYREF
  __int64 v77; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v78; // [rsp+A8h] [rbp-88h] BYREF
  __int64 v79; // [rsp+B0h] [rbp-80h]
  __int64 v80; // [rsp+B8h] [rbp-78h]
  unsigned __int64 v81; // [rsp+C0h] [rbp-70h]
  unsigned __int64 v82; // [rsp+C8h] [rbp-68h]
  __int64 v83; // [rsp+D0h] [rbp-60h]
  __int64 v84; // [rsp+D8h] [rbp-58h]
  __int64 v85; // [rsp+E0h] [rbp-50h]
  __int64 v86; // [rsp+E8h] [rbp-48h]
  __int64 v87; // [rsp+F0h] [rbp-40h]
  __int64 v88; // [rsp+F8h] [rbp-38h]

  v23 = a3;
  v24 = a5;
  v25 = a4;
  v26 = a1;
  v27 = a2;
  v28 = a22;
  if ( a22 )
  {
LABEL_4:
    v75 = v24;
    v73 = &v75;
    v79 = a7;
    v76 = v27;
    v80 = a17;
    v77 = v23;
    v81 = a19;
    v78 = v25;
    v82 = a20;
    v83 = a11;
    v84 = a18;
    v85 = a21;
    v74 = 0xB0000000BLL;
    if ( a21 )
    {
      v31 = 11;
      v32 = 11;
    }
    else if ( a18 )
    {
      LODWORD(v74) = 10;
      v31 = 10;
      v32 = 10;
    }
    else if ( a11 )
    {
      LODWORD(v74) = 9;
      v31 = 9;
      v32 = 9;
    }
    else
    {
      LODWORD(v74) = 8;
      v31 = 8;
      v32 = 8;
    }
    v67 = v31;
    v33 = *v26 + 880LL;
    v34 = sub_161E980(48, v32);
    v35 = v34;
    if ( v34 )
    {
      sub_1623D80(v34, (_DWORD)v26, 17, v28, (unsigned int)&v75, v67, 0, 0);
      *(_WORD *)(v35 + 2) = 46;
      *(_DWORD *)(v35 + 24) = a6;
      *(_DWORD *)(v35 + 28) = a10;
      *(_DWORD *)(v35 + 32) = a13;
      *(_DWORD *)(v35 + 36) = a14;
      *(_BYTE *)(v35 + 40) = (16 * (a16 & 1))
                           | (8 * (a9 & 1))
                           | (4 * (a8 & 1)) & 0xE7
                           | a12 & 3
                           | *(_BYTE *)(v35 + 40) & 0xE0;
      *(_DWORD *)(v35 + 44) = a15;
    }
    result = sub_15BFB90(v35, v28, v33);
    if ( v73 != &v75 )
    {
      v72 = result;
      _libc_free((unsigned __int64)v73);
      return v72;
    }
    return result;
  }
  v73 = (__int64 *)a2;
  v29 = *a1;
  v74 = a3;
  LODWORD(v77) = a6;
  v75 = a4;
  v78 = a7;
  v76 = a5;
  LOBYTE(v79) = a8;
  BYTE1(v79) = a9;
  HIDWORD(v79) = a10;
  v80 = a11;
  v81 = __PAIR64__(a13, a12);
  v82 = __PAIR64__(a15, a14);
  LOBYTE(v83) = a16;
  v84 = a17;
  v85 = a18;
  v86 = a19;
  v87 = a20;
  v88 = a21;
  v66 = *(_QWORD *)(v29 + 888);
  if ( !*(_DWORD *)(v29 + 904) )
    goto LABEL_3;
  if ( a4 != 0 && a2 != 0 && a9 != 1 && *(_BYTE *)a2 == 13 && *(_QWORD *)(a2 + 8 * (7LL - *(unsigned int *)(a2 + 8))) )
  {
    v68 = *(_DWORD *)(v29 + 904);
    v36 = sub_15B2D00(&v75, (__int64 *)&v73);
    v37 = v68;
    v23 = a3;
    v24 = a5;
    v38 = v36;
  }
  else
  {
    v69 = *(_DWORD *)(v29 + 904);
    v39 = sub_15B55D0(&v74, (__int64 *)&v73, &v76, &v78, (int *)&v77);
    v24 = a5;
    v23 = a3;
    v37 = v69;
    v38 = v39;
  }
  v40 = v37 - 1;
  v41 = v40 & v38;
  v42 = *(_QWORD *)(v66 + 8 * v41);
  v65 = *(_QWORD *)(v29 + 888);
  v70 = *(_DWORD *)(v29 + 904);
  if ( v42 == -8 )
    goto LABEL_3;
  v43 = (__int64)v73;
  v57 = 1;
  v44 = v40 & v38;
  v55 = v85;
  v59 = v27;
  v56 = v75;
  v58 = v25;
  v45 = (__int64 *)(v66 + 8 * v41);
  v46 = v40;
  v47 = BYTE1(v79) | (v73 == 0 || v75 == 0);
  while ( 1 )
  {
    if ( v42 != -16 )
    {
      if ( !v47 && *(_BYTE *)v43 == 13 )
      {
        if ( *(_QWORD *)(v43 + 8 * (7LL - *(unsigned int *)(v43 + 8))) )
        {
          if ( (*(_BYTE *)(v42 + 40) & 8) == 0 )
          {
            v48 = *(unsigned int *)(v42 + 8);
            if ( v43 == *(_QWORD *)(v42 + 8 * (1 - v48)) )
            {
              v49 = *(_QWORD *)(v42 + 8 * (3 - v48));
              if ( v49 )
              {
                if ( v56 == v49 )
                {
                  v50 = 0;
                  if ( (unsigned int)v48 > 9 )
                    v50 = *(_QWORD *)(v42 + 8 * (9 - v48));
                  if ( v55 == v50 )
                    break;
                }
              }
            }
          }
        }
      }
      v53 = v24;
      v54 = v23;
      v52 = sub_15AFB30((__int64)&v73, v42);
      v23 = v54;
      v24 = v53;
      if ( v52 )
        break;
    }
    v44 = v46 & (v57 + v44);
    v45 = (__int64 *)(v66 + 8LL * v44);
    v42 = *v45;
    if ( *v45 == -8 )
    {
      v26 = a1;
      v27 = v59;
      v25 = v58;
      v28 = 0;
      goto LABEL_3;
    }
    ++v57;
  }
  v51 = v45;
  v26 = a1;
  v27 = v59;
  v25 = v58;
  v28 = 0;
  if ( v51 == (__int64 *)(v65 + 8LL * v70) || (result = *v51) == 0 )
  {
LABEL_3:
    result = 0;
    if ( !a23 )
      return result;
    goto LABEL_4;
  }
  return result;
}
