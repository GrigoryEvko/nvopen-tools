// Function: sub_B0B820
// Address: 0xb0b820
//
__int64 __fastcall sub_B0B820(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __int64 a7,
        char a8,
        char a9,
        __int64 a10,
        __int64 a11,
        int a12,
        __int64 a13,
        unsigned int a14,
        char a15)
{
  __int64 v15; // r10
  __int64 v16; // r14
  int v17; // r13d
  _QWORD *v18; // r12
  __int64 v19; // rbx
  unsigned int v20; // r15d
  __int64 v21; // rax
  int v22; // eax
  int v23; // ecx
  int v24; // r8d
  int v26; // r9d
  unsigned int v27; // r11d
  unsigned int v28; // r10d
  __int64 v29; // r15
  __int64 v30; // r14
  __int64 *v31; // r12
  __int64 v32; // rbx
  unsigned __int8 v33; // al
  _QWORD *v34; // rsi
  __int64 v35; // rax
  unsigned int v36; // r11d
  __int64 result; // rax
  __int64 v38; // r14
  __int64 v39; // rbx
  __int64 v40; // rax
  _BYTE *v41; // rax
  _BYTE *v42; // rax
  _BYTE *v43; // rax
  _BYTE *v44; // rax
  _BYTE *v45; // rax
  __int64 *v46; // rdi
  __int64 v47; // rcx
  int v48; // [rsp+Ch] [rbp-D4h]
  int v49; // [rsp+10h] [rbp-D0h]
  unsigned int v50; // [rsp+14h] [rbp-CCh]
  __int64 v51; // [rsp+18h] [rbp-C8h]
  _BYTE *v53; // [rsp+20h] [rbp-C0h]
  int v54; // [rsp+28h] [rbp-B8h]
  __int64 v55; // [rsp+28h] [rbp-B8h]
  __int64 v56; // [rsp+30h] [rbp-B0h]
  int v57; // [rsp+38h] [rbp-A8h]
  __int64 v59; // [rsp+50h] [rbp-90h] BYREF
  __int64 v60; // [rsp+58h] [rbp-88h] BYREF
  __int64 v61; // [rsp+60h] [rbp-80h] BYREF
  __int64 v62; // [rsp+68h] [rbp-78h] BYREF
  __int64 v63; // [rsp+70h] [rbp-70h] BYREF
  __int64 v64; // [rsp+78h] [rbp-68h] BYREF
  __int64 v65; // [rsp+80h] [rbp-60h] BYREF
  __int64 v66; // [rsp+88h] [rbp-58h] BYREF
  __int64 v67; // [rsp+90h] [rbp-50h]
  int v68; // [rsp+98h] [rbp-48h]
  __int64 v69[8]; // [rsp+A0h] [rbp-40h] BYREF
  unsigned int v70; // [rsp+128h] [rbp+48h]

  v15 = a5;
  v16 = a4;
  v17 = a6;
  v18 = a1;
  v19 = a3;
  v20 = a14;
  if ( a14 )
    goto LABEL_13;
  v21 = *a1;
  v60 = a3;
  v61 = a4;
  v59 = a2;
  v64 = a7;
  LOBYTE(v65) = a8;
  BYTE1(v65) = a9;
  v62 = a5;
  LODWORD(v63) = a6;
  v66 = a10;
  v67 = a11;
  v68 = a12;
  v69[0] = a13;
  v51 = v21;
  v54 = *(_DWORD *)(v21 + 1296);
  v56 = *(_QWORD *)(v21 + 1280);
  if ( v54 )
  {
    v22 = sub_AF8D50(&v59, &v60, &v61, &v62, (int *)&v63, &v64, (__int8 *)&v65, (__int8 *)&v65 + 1, &v66, v69);
    v23 = v17;
    v24 = 1;
    v26 = v54 - 1;
    v55 = a5;
    v27 = v26 & v22;
    v28 = 0;
    v29 = v16;
    v30 = v19;
    while ( 1 )
    {
      v31 = (__int64 *)(v56 + 8LL * v27);
      v32 = *v31;
      if ( *v31 == -4096 )
      {
        v19 = v30;
        v16 = v29;
        v20 = v28;
        v15 = v55;
        v18 = a1;
        v17 = v23;
        goto LABEL_12;
      }
      if ( v32 != -8192 )
      {
        v53 = (_BYTE *)(v32 - 16);
        v33 = *(_BYTE *)(v32 - 16);
        v34 = (v33 & 2) != 0 ? *(_QWORD **)(v32 - 32) : &v53[-8 * ((v33 >> 2) & 0xF)];
        if ( v59 == *v34 )
        {
          v70 = v28;
          v48 = v23;
          v49 = v24;
          v50 = v27;
          v57 = v26;
          v35 = sub_AF5140(v32, 1u);
          v28 = v70;
          v26 = v57;
          v27 = v50;
          v24 = v49;
          v23 = v48;
          if ( v60 == v35 )
          {
            v40 = sub_AF5140(v32, 5u);
            v28 = v70;
            v26 = v57;
            v27 = v50;
            v24 = v49;
            v23 = v48;
            if ( v61 == v40 )
            {
              v41 = sub_A17150(v53);
              v26 = v57;
              v28 = v70;
              v27 = v50;
              v24 = v49;
              v23 = v48;
              if ( v62 == *((_QWORD *)v41 + 2) && (_DWORD)v63 == *(_DWORD *)(v32 + 16) )
              {
                v42 = sub_A17150(v53);
                v26 = v57;
                v28 = v70;
                v27 = v50;
                v24 = v49;
                v23 = v48;
                if ( v64 == *((_QWORD *)v42 + 3) && (_WORD)v65 == *(_WORD *)(v32 + 20) )
                {
                  v43 = sub_A17150(v53);
                  v26 = v57;
                  v28 = v70;
                  v27 = v50;
                  v24 = v49;
                  v23 = v48;
                  if ( v66 == *((_QWORD *)v43 + 6) )
                  {
                    v44 = sub_A17150(v53);
                    v26 = v57;
                    v28 = v70;
                    v27 = v50;
                    v24 = v49;
                    v23 = v48;
                    if ( v67 == *((_QWORD *)v44 + 7) && v68 == *(_DWORD *)(v32 + 4) )
                    {
                      v45 = sub_A17150(v53);
                      v26 = v57;
                      v28 = v70;
                      v27 = v50;
                      v24 = v49;
                      v23 = v48;
                      if ( v69[0] == *((_QWORD *)v45 + 8) )
                        break;
                    }
                  }
                }
              }
            }
          }
        }
      }
      v36 = v24 + v27;
      ++v24;
      v27 = v26 & v36;
    }
    v46 = v31;
    v18 = a1;
    v17 = v48;
    v47 = v32;
    v19 = v30;
    v16 = v29;
    v20 = v70;
    v15 = v55;
    if ( v46 != (__int64 *)(*(_QWORD *)(v51 + 1280) + 8LL * *(unsigned int *)(v51 + 1296)) )
      return v47;
  }
LABEL_12:
  result = 0;
  if ( a15 )
  {
LABEL_13:
    v63 = v19;
    v60 = v19;
    v59 = a2;
    v64 = v16;
    v62 = a7;
    v61 = v15;
    v65 = a10;
    v66 = a11;
    v67 = a13;
    v38 = *v18 + 1272LL;
    v39 = sub_B97910(24, 9, v20);
    if ( v39 )
    {
      sub_AF3F90(v39, (int)v18, 25, v20, v17, a12, (__int64)&v59, 9);
      *(_BYTE *)(v39 + 20) = a8;
      *(_BYTE *)(v39 + 21) = a9;
    }
    return sub_B0B740(v39, v20, v38);
  }
  return result;
}
