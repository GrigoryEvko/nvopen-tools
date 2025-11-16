// Function: sub_1173460
// Address: 0x1173460
//
_QWORD *__fastcall sub_1173460(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 *v4; // rdi
  __int64 v5; // rbx
  __int64 v6; // rsi
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 *v9; // rdx
  __int64 v10; // rcx
  __int64 *v11; // r15
  __int64 *v12; // r12
  const void **v13; // r14
  _QWORD *v14; // r15
  __int64 v16; // rdx
  size_t v17; // rdx
  __int64 v18; // rdi
  const char *v19; // rax
  int v20; // r15d
  int v21; // r15d
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // r14
  __int64 v25; // rsi
  __int64 v26; // r8
  _QWORD *v27; // r15
  _QWORD *v28; // rcx
  __int64 v29; // rdx
  __int64 v30; // rax
  _QWORD *v31; // rdx
  _QWORD *v32; // r8
  _QWORD *v33; // r9
  __int64 v34; // rsi
  __int64 v35; // r15
  int v36; // eax
  int v37; // eax
  unsigned int v38; // edi
  __int64 v39; // rax
  __int64 v40; // rdi
  __int64 v41; // rdi
  __int64 v42; // rax
  __int64 v43; // rdx
  unsigned int *v44; // r11
  __int64 v45; // rbx
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  int v49; // eax
  _QWORD *v50; // [rsp+0h] [rbp-90h]
  _QWORD *v51; // [rsp+8h] [rbp-88h]
  _QWORD *v52; // [rsp+10h] [rbp-80h]
  _QWORD *v53; // [rsp+18h] [rbp-78h]
  unsigned int *v54; // [rsp+20h] [rbp-70h]
  __int64 v55; // [rsp+28h] [rbp-68h]
  __int64 v56; // [rsp+28h] [rbp-68h]
  const char *v57; // [rsp+30h] [rbp-60h] BYREF
  __int64 v58; // [rsp+38h] [rbp-58h]
  const char *v59; // [rsp+40h] [rbp-50h]
  __int16 v60; // [rsp+50h] [rbp-40h]

  v3 = a2;
  v4 = *(__int64 **)(a2 - 8);
  v5 = *v4;
  v6 = 4LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(v3 + 7) & 0x40) != 0 )
  {
    v7 = (__int64)&v4[v6];
  }
  else
  {
    v4 = (__int64 *)(v3 - v6 * 8);
    v7 = v3;
  }
  v8 = sub_116D080((__int64)v4, v7, 1);
  if ( v9 == (__int64 *)v8 )
  {
    v18 = *(_QWORD *)(v5 - 32);
  }
  else
  {
    v10 = v3;
    v11 = (__int64 *)v8;
    v12 = v9;
    do
    {
      v13 = (const void **)*v11;
      if ( *(_BYTE *)*v11 != 93 )
        return 0;
      v55 = v10;
      if ( !(unsigned __int8)sub_BD36B0(*v11) )
        return 0;
      v16 = *((unsigned int *)v13 + 20);
      if ( *(_DWORD *)(v5 + 80) != (_DWORD)v16 )
        return 0;
      v17 = 4 * v16;
      v10 = v55;
      if ( v17 )
      {
        v49 = memcmp(v13[9], *(const void **)(v5 + 72), v17);
        v10 = v55;
        if ( v49 )
          return 0;
      }
      v18 = *(_QWORD *)(v5 - 32);
      if ( *(_QWORD *)(v18 + 8) != *((_QWORD *)*(v13 - 4) + 1) )
        return 0;
      v11 += 4;
    }
    while ( v12 != v11 );
    v3 = v10;
  }
  v19 = sub_BD5D20(v18);
  v20 = *(_DWORD *)(v3 + 4);
  v57 = v19;
  v60 = 773;
  v21 = v20 & 0x7FFFFFF;
  v58 = v22;
  v59 = ".pn";
  v56 = *(_QWORD *)(*(_QWORD *)(v5 - 32) + 8LL);
  v23 = sub_BD2DA0(80);
  v24 = v23;
  if ( v23 )
  {
    sub_B44260(v23, v56, 55, 0x8000000u, 0, 0);
    *(_DWORD *)(v24 + 72) = v21;
    sub_BD6B50((unsigned __int8 *)v24, &v57);
    sub_BD2A10(v24, *(_DWORD *)(v24 + 72), 1);
  }
  v25 = *(_QWORD *)(v3 - 8);
  v26 = 32LL * (*(_DWORD *)(v3 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(v3 + 7) & 0x40) != 0 )
  {
    v27 = (_QWORD *)(v25 + v26);
    v28 = *(_QWORD **)(v3 - 8);
  }
  else
  {
    v27 = (_QWORD *)v3;
    v28 = (_QWORD *)(v3 - v26);
  }
  v29 = 32LL * *(unsigned int *)(v3 + 72);
  v30 = v29 + 8LL * (*(_DWORD *)(v3 + 4) & 0x7FFFFFF);
  v31 = (_QWORD *)(v25 + v29);
  v32 = (_QWORD *)(v25 + v30);
  if ( (_QWORD *)(v25 + v30) != v31 )
  {
    v33 = v27;
    if ( v28 != v27 )
    {
      do
      {
        v34 = *v31;
        v35 = *(_QWORD *)(*v28 - 32LL);
        v36 = *(_DWORD *)(v24 + 4) & 0x7FFFFFF;
        if ( v36 == *(_DWORD *)(v24 + 72) )
        {
          v50 = v33;
          v51 = v28;
          v52 = v31;
          v53 = v32;
          sub_B48D90(v24);
          v33 = v50;
          v28 = v51;
          v31 = v52;
          v32 = v53;
          v36 = *(_DWORD *)(v24 + 4) & 0x7FFFFFF;
        }
        v37 = (v36 + 1) & 0x7FFFFFF;
        v38 = v37 | *(_DWORD *)(v24 + 4) & 0xF8000000;
        v39 = *(_QWORD *)(v24 - 8) + 32LL * (unsigned int)(v37 - 1);
        *(_DWORD *)(v24 + 4) = v38;
        if ( *(_QWORD *)v39 )
        {
          v40 = *(_QWORD *)(v39 + 8);
          **(_QWORD **)(v39 + 16) = v40;
          if ( v40 )
            *(_QWORD *)(v40 + 16) = *(_QWORD *)(v39 + 16);
        }
        *(_QWORD *)v39 = v35;
        if ( v35 )
        {
          v41 = *(_QWORD *)(v35 + 16);
          *(_QWORD *)(v39 + 8) = v41;
          if ( v41 )
            *(_QWORD *)(v41 + 16) = v39 + 8;
          *(_QWORD *)(v39 + 16) = v35 + 16;
          *(_QWORD *)(v35 + 16) = v39;
        }
        v28 += 4;
        ++v31;
        *(_QWORD *)(*(_QWORD *)(v24 - 8)
                  + 32LL * *(unsigned int *)(v24 + 72)
                  + 8LL * ((*(_DWORD *)(v24 + 4) & 0x7FFFFFFu) - 1)) = v34;
      }
      while ( v28 != v33 && v32 != v31 );
    }
  }
  sub_B44220((_QWORD *)v24, v3 + 24, 0);
  v42 = *(_QWORD *)(a1 + 40);
  v57 = (const char *)v24;
  sub_11715E0(v42 + 2096, (__int64 *)&v57);
  v57 = sub_BD5D20(v3);
  v60 = 261;
  v58 = v43;
  v44 = *(unsigned int **)(v5 + 72);
  v45 = *(unsigned int *)(v5 + 80);
  v54 = v44;
  v14 = sub_BD2C40(104, unk_3F10A14);
  if ( v14 )
  {
    v46 = sub_B501B0(*(_QWORD *)(v24 + 8), v54, v45);
    sub_B44260((__int64)v14, v46, 64, 1u, 0, 0);
    if ( *(v14 - 4) )
    {
      v47 = *(v14 - 3);
      *(_QWORD *)*(v14 - 2) = v47;
      if ( v47 )
        *(_QWORD *)(v47 + 16) = *(v14 - 2);
    }
    *(v14 - 4) = v24;
    v48 = *(_QWORD *)(v24 + 16);
    *(v14 - 3) = v48;
    if ( v48 )
      *(_QWORD *)(v48 + 16) = v14 - 3;
    *(v14 - 2) = v24 + 16;
    *(_QWORD *)(v24 + 16) = v14 - 4;
    v14[9] = v14 + 11;
    v14[10] = 0x400000000LL;
    sub_B50030((__int64)v14, v54, v45, (__int64)&v57);
  }
  sub_116D800(a1, (__int64)v14, v3);
  return v14;
}
