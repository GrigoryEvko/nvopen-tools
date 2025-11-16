// Function: sub_B0C150
// Address: 0xb0c150
//
__int64 __fastcall sub_B0C150(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        int a5,
        __int64 a6,
        int a7,
        int a8,
        int a9,
        __int64 a10,
        unsigned int a11,
        char a12)
{
  __int64 v12; // r10
  _QWORD *v15; // r12
  int v16; // ebx
  unsigned int v17; // r15d
  __int64 v18; // rax
  int v19; // eax
  int v20; // ecx
  int v21; // r8d
  int v22; // r9d
  unsigned int v23; // r11d
  unsigned int v24; // r10d
  __int64 *v25; // r12
  __int64 v26; // rbx
  unsigned __int8 v27; // al
  _QWORD *v28; // rsi
  __int64 v29; // rax
  unsigned int v30; // r11d
  __int64 result; // rax
  __int64 v32; // r13
  __int64 v33; // r14
  _BYTE *v34; // rax
  _BYTE *v35; // rax
  _BYTE *v36; // rax
  __int64 *v37; // rsi
  int v38; // [rsp+4h] [rbp-ACh]
  int v39; // [rsp+8h] [rbp-A8h]
  unsigned int v40; // [rsp+Ch] [rbp-A4h]
  __int64 v41; // [rsp+10h] [rbp-A0h]
  _BYTE *v43; // [rsp+18h] [rbp-98h]
  int v44; // [rsp+20h] [rbp-90h]
  __int64 v45; // [rsp+20h] [rbp-90h]
  __int64 v46; // [rsp+28h] [rbp-88h]
  int v47; // [rsp+30h] [rbp-80h]
  __int64 v49; // [rsp+40h] [rbp-70h] BYREF
  __int64 v50; // [rsp+48h] [rbp-68h] BYREF
  __int64 v51; // [rsp+50h] [rbp-60h] BYREF
  __int64 v52; // [rsp+58h] [rbp-58h] BYREF
  __int64 v53; // [rsp+60h] [rbp-50h] BYREF
  int v54; // [rsp+68h] [rbp-48h] BYREF
  int v55; // [rsp+6Ch] [rbp-44h] BYREF
  int v56; // [rsp+70h] [rbp-40h]
  __int64 v57[7]; // [rsp+78h] [rbp-38h] BYREF
  unsigned int v58; // [rsp+E0h] [rbp+30h]

  v12 = a6;
  v15 = a1;
  v16 = a5;
  v17 = a11;
  if ( a11 )
  {
LABEL_13:
    v50 = a3;
    v51 = a4;
    v49 = a2;
    v52 = v12;
    v53 = a10;
    v32 = *v15 + 1304LL;
    v33 = sub_B97910(32, 5, v17);
    if ( v33 )
    {
      sub_AF3F90(v33, (int)v15, 26, v17, v16, a9, (__int64)&v49, 5);
      *(_WORD *)(v33 + 20) = a7;
      *(_DWORD *)(v33 + 24) = a8;
    }
    return sub_B0C070(v33, v17, v32);
  }
  v18 = *a1;
  v50 = a3;
  v51 = a4;
  v49 = a2;
  LODWORD(v52) = a5;
  v53 = a6;
  v54 = a7;
  v55 = a8;
  v56 = a9;
  v57[0] = a10;
  v41 = v18;
  v44 = *(_DWORD *)(v18 + 1328);
  v46 = *(_QWORD *)(v18 + 1312);
  if ( !v44 )
    goto LABEL_12;
  v19 = sub_AF8A50(&v49, &v50, &v51, (int *)&v52, &v53, &v54, &v55, v57);
  v20 = v16;
  v21 = 1;
  v22 = v44 - 1;
  v45 = a6;
  v23 = v22 & v19;
  v24 = 0;
  while ( 1 )
  {
    v25 = (__int64 *)(v46 + 8LL * v23);
    v26 = *v25;
    if ( *v25 == -4096 )
    {
      v15 = a1;
      v17 = v24;
      v12 = v45;
      v16 = v20;
      goto LABEL_12;
    }
    if ( v26 != -8192 )
    {
      v43 = (_BYTE *)(v26 - 16);
      v27 = *(_BYTE *)(v26 - 16);
      v28 = (v27 & 2) != 0 ? *(_QWORD **)(v26 - 32) : &v43[-8 * ((v27 >> 2) & 0xF)];
      if ( v49 == *v28 )
      {
        v58 = v24;
        v38 = v20;
        v39 = v21;
        v40 = v23;
        v47 = v22;
        v29 = sub_AF5140(v26, 1u);
        v22 = v47;
        v23 = v40;
        v21 = v39;
        v20 = v38;
        v24 = v58;
        if ( v50 == v29 )
        {
          v34 = sub_A17150(v43);
          v22 = v47;
          v24 = v58;
          v23 = v40;
          v21 = v39;
          v20 = v38;
          if ( v51 == *((_QWORD *)v34 + 2) && (_DWORD)v52 == *(_DWORD *)(v26 + 16) )
          {
            v35 = sub_A17150(v43);
            v22 = v47;
            v24 = v58;
            v23 = v40;
            v21 = v39;
            v20 = v38;
            if ( v53 == *((_QWORD *)v35 + 3)
              && v54 == *(unsigned __int16 *)(v26 + 20)
              && v55 == *(_DWORD *)(v26 + 24)
              && v56 == *(_DWORD *)(v26 + 4) )
            {
              v36 = sub_A17150(v43);
              v22 = v47;
              v24 = v58;
              v23 = v40;
              v21 = v39;
              v20 = v38;
              if ( v57[0] == *((_QWORD *)v36 + 4) )
                break;
            }
          }
        }
      }
    }
    v30 = v21 + v23;
    ++v21;
    v23 = v22 & v30;
  }
  result = v26;
  v16 = v38;
  v37 = v25;
  v15 = a1;
  v17 = v58;
  v12 = v45;
  if ( v37 == (__int64 *)(*(_QWORD *)(v41 + 1312) + 8LL * *(unsigned int *)(v41 + 1328)) )
  {
LABEL_12:
    result = 0;
    if ( !a12 )
      return result;
    goto LABEL_13;
  }
  return result;
}
