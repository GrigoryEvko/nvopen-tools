// Function: sub_B0A3E0
// Address: 0xb0a3e0
//
_BYTE *__fastcall sub_B0A3E0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        int a8,
        char a9,
        unsigned int a10,
        char a11)
{
  __int64 v11; // r10
  __int64 v13; // rbx
  unsigned int v14; // r14d
  __int64 v15; // r11
  int v16; // r15d
  int v17; // eax
  unsigned int v18; // edx
  __int64 v19; // r11
  __int64 *v20; // rbx
  __int64 v21; // rcx
  unsigned __int8 v22; // al
  __int64 *v23; // r15
  _BYTE *result; // rax
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // r9
  _BYTE *v28; // r15
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // [rsp+8h] [rbp-C8h]
  __int64 v34; // [rsp+10h] [rbp-C0h]
  int v35; // [rsp+18h] [rbp-B8h]
  int v36; // [rsp+1Ch] [rbp-B4h]
  __int64 v37; // [rsp+20h] [rbp-B0h]
  __int64 v38; // [rsp+20h] [rbp-B0h]
  __int64 v39; // [rsp+28h] [rbp-A8h]
  __int64 v40; // [rsp+28h] [rbp-A8h]
  __int64 v42; // [rsp+30h] [rbp-A0h]
  __int64 v43; // [rsp+38h] [rbp-98h]
  __int64 v44; // [rsp+38h] [rbp-98h]
  __int64 v45; // [rsp+50h] [rbp-80h]
  unsigned int v46; // [rsp+58h] [rbp-78h]
  __int64 v47; // [rsp+60h] [rbp-70h] BYREF
  __int64 v48; // [rsp+68h] [rbp-68h] BYREF
  __int64 v49; // [rsp+70h] [rbp-60h] BYREF
  __int64 v50; // [rsp+78h] [rbp-58h] BYREF
  __int64 v51; // [rsp+80h] [rbp-50h] BYREF
  __int64 v52; // [rsp+88h] [rbp-48h]
  int v53; // [rsp+90h] [rbp-40h]
  char v54; // [rsp+94h] [rbp-3Ch]
  unsigned int v55; // [rsp+F8h] [rbp+28h]

  v11 = a4;
  v13 = a2;
  v14 = a10;
  if ( a10 )
    goto LABEL_12;
  v15 = *a1;
  v47 = a2;
  v48 = a3;
  v52 = a7;
  v49 = a4;
  v53 = a8;
  v50 = a5;
  v51 = a6;
  v54 = a9;
  v16 = *(_DWORD *)(v15 + 1200);
  v43 = v15;
  v45 = *(_QWORD *)(v15 + 1184);
  if ( v16 )
  {
    v37 = a6;
    v39 = a5;
    v17 = sub_AFBE30(&v48, &v49, &v50, &v51);
    v18 = 0;
    v36 = 1;
    v35 = v16 - 1;
    v19 = v43;
    v46 = (v16 - 1) & v17;
    v11 = a4;
    a5 = v39;
    a6 = v37;
    while ( 1 )
    {
      v20 = (__int64 *)(v45 + 8LL * v46);
      v21 = *v20;
      if ( *v20 == -4096 )
      {
        v13 = a2;
        v14 = v18;
        goto LABEL_11;
      }
      if ( v21 != -8192 )
      {
        v22 = *(_BYTE *)(v21 - 16);
        v23 = (v22 & 2) != 0 ? *(__int64 **)(v21 - 32) : (__int64 *)(v21 - 16 - 8LL * ((v22 >> 2) & 0xF));
        if ( v48 == v23[1] )
        {
          v55 = v18;
          v34 = a6;
          v38 = a5;
          v40 = v11;
          v42 = v19;
          v33 = *v20;
          v44 = *v20;
          v29 = sub_AF5140(*v20, 2u);
          v18 = v55;
          v19 = v42;
          v11 = v40;
          a5 = v38;
          a6 = v34;
          if ( v49 == v29 )
          {
            v30 = sub_AF5140(v44, 3u);
            v18 = v55;
            v19 = v42;
            v11 = v40;
            a5 = v38;
            a6 = v34;
            if ( v50 == v30 )
            {
              v31 = sub_AF5140(v44, 4u);
              v18 = v55;
              v19 = v42;
              v11 = v40;
              a5 = v38;
              a6 = v34;
              if ( v51 == v31 )
              {
                v32 = sub_AF5140(v44, 5u);
                v18 = v55;
                v19 = v42;
                v11 = v40;
                a5 = v38;
                a6 = v34;
                if ( v52 == v32 )
                {
                  if ( *(_BYTE *)v44 != 16 )
                    v33 = *v23;
                  if ( v47 == v33 && v53 == *(_DWORD *)(v44 + 4) && v54 == (unsigned __int8)BYTE1(*(_QWORD *)v44) >> 7 )
                    break;
                }
              }
            }
          }
        }
      }
      v46 = v35 & (v36 + v46);
      ++v36;
    }
    v13 = a2;
    v14 = v55;
    if ( v45 + 8LL * v46 != *(_QWORD *)(v42 + 1184) + 8LL * *(unsigned int *)(v42 + 1200) )
      return (_BYTE *)v44;
  }
LABEL_11:
  result = 0;
  if ( a11 )
  {
LABEL_12:
    v47 = v13;
    v48 = a3;
    v52 = a7;
    v25 = *a1 + 1176;
    v49 = v11;
    v50 = a5;
    v51 = a6;
    v26 = sub_B97910(16, 6, v14);
    v28 = (_BYTE *)v26;
    if ( v26 )
      sub_AF3EE0(v26, (int)a1, v14, a8, a9, v27, (__int64)&v47, 6);
    return sub_B0A300(v28, v14, v25);
  }
  return result;
}
