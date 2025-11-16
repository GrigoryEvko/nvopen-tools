// Function: sub_1172FC0
// Address: 0x1172fc0
//
__int64 **__fastcall sub_1172FC0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  unsigned __int64 v3; // rax
  int v4; // eax
  unsigned __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // r9
  unsigned __int64 *v9; // r15
  unsigned __int64 *v10; // rbx
  unsigned __int64 *v11; // rax
  __int64 **v12; // r14
  unsigned __int64 *v14; // r13
  _BYTE *v15; // rbx
  unsigned __int8 v16; // al
  __int64 v17; // r12
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  const char *v23; // rdx
  __int64 v24; // rax
  __int64 v25; // r15
  __int64 v26; // rbx
  int v27; // eax
  unsigned int v28; // esi
  __int64 v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rsi
  __int64 v32; // rdx
  __int64 v33; // r14
  int v34; // eax
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rax
  __int64 v38; // rbx
  unsigned __int64 v39; // rdx
  __int64 v40; // [rsp+0h] [rbp-D0h]
  __int64 v41; // [rsp+8h] [rbp-C8h]
  int v42; // [rsp+1Ch] [rbp-B4h]
  unsigned int v43; // [rsp+20h] [rbp-B0h]
  unsigned int v44; // [rsp+30h] [rbp-A0h]
  __int64 v45; // [rsp+30h] [rbp-A0h]
  const char *v46[4]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v47; // [rsp+60h] [rbp-70h]
  _BYTE *v48; // [rsp+70h] [rbp-60h] BYREF
  __int64 v49; // [rsp+78h] [rbp-58h]
  _BYTE v50[80]; // [rsp+80h] [rbp-50h] BYREF

  v2 = *(_QWORD *)(a2 + 40);
  v3 = *(_QWORD *)(v2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v3 != v2 + 48 )
  {
    if ( !v3 )
      BUG();
    v4 = *(unsigned __int8 *)(v3 - 24);
    if ( (unsigned int)(v4 - 30) <= 0xA )
    {
      v5 = (unsigned int)(v4 - 39);
      if ( (unsigned int)v5 <= 0x38 )
      {
        v6 = 0x100060000000001LL;
        if ( _bittest64(&v6, v5) )
          return 0;
      }
    }
  }
  v7 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v42 = v7;
  if ( (unsigned int)v7 <= 2 )
    return 0;
  v41 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v8 = 4 * v7;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v10 = *(unsigned __int64 **)(a2 - 8);
    v9 = &v10[v8];
  }
  else
  {
    v9 = (unsigned __int64 *)a2;
    v10 = (unsigned __int64 *)(a2 - v8 * 8);
  }
  v11 = v10;
  while ( *(_BYTE *)*v11 != 68 )
  {
    v11 += 4;
    if ( v9 == v11 )
      return 0;
  }
  v12 = *(__int64 ***)(*(_QWORD *)(*v11 - 32) + 8LL);
  if ( v12 )
  {
    v40 = a2;
    v14 = v10;
    v48 = v50;
    v49 = 0x400000000LL;
    v44 = 0;
    v43 = 0;
    do
    {
      v15 = (_BYTE *)*v14;
      v16 = *(_BYTE *)*v14;
      if ( v16 <= 0x1Cu )
      {
        if ( v16 > 0x15u )
          goto LABEL_19;
        v17 = sub_AD4C30(*v14, v12, 0);
        a2 = v17;
        v18 = sub_96F480(0x27u, v17, *((_QWORD *)v15 + 1), *(_QWORD *)(a1 + 88));
        if ( v15 != (_BYTE *)v18 || v18 == 0 || !v17 )
          goto LABEL_19;
        v21 = (unsigned int)v49;
        v22 = (unsigned int)v49 + 1LL;
        if ( v22 > HIDWORD(v49) )
        {
          a2 = (__int64)v50;
          sub_C8D5F0((__int64)&v48, v50, v22, 8u, v19, v20);
          v21 = (unsigned int)v49;
        }
        ++v44;
        *(_QWORD *)&v48[8 * v21] = v17;
        LODWORD(v49) = v49 + 1;
      }
      else
      {
        if ( v16 != 68 || v12 != *(__int64 ***)(*((_QWORD *)v15 - 4) + 8LL) || !(unsigned __int8)sub_BD36B0(*v14) )
          goto LABEL_19;
        v37 = (unsigned int)v49;
        v38 = *((_QWORD *)v15 - 4);
        v39 = (unsigned int)v49 + 1LL;
        if ( v39 > HIDWORD(v49) )
        {
          a2 = (__int64)v50;
          sub_C8D5F0((__int64)&v48, v50, v39, 8u, v35, v36);
          v37 = (unsigned int)v49;
        }
        ++v43;
        *(_QWORD *)&v48[8 * v37] = v38;
        LODWORD(v49) = v49 + 1;
      }
      v14 += 4;
    }
    while ( v9 != v14 );
    a2 = v44;
    if ( !v44 || v43 <= 1 )
    {
LABEL_19:
      v12 = 0;
      goto LABEL_20;
    }
    v46[0] = sub_BD5D20(v40);
    v47 = 773;
    v46[1] = v23;
    v46[2] = ".shrunk";
    v24 = sub_BD2DA0(80);
    v25 = v24;
    if ( v24 )
    {
      sub_B44260(v24, (__int64)v12, 55, 0x8000000u, 0, 0);
      *(_DWORD *)(v25 + 72) = v42;
      sub_BD6B50((unsigned __int8 *)v25, v46);
      sub_BD2A10(v25, *(_DWORD *)(v25 + 72), 1);
    }
    v26 = 0;
    do
    {
      v32 = *(_QWORD *)(*(_QWORD *)(v40 - 8) + 32LL * *(unsigned int *)(v40 + 72) + v26);
      v33 = *(_QWORD *)&v48[v26];
      v34 = *(_DWORD *)(v25 + 4) & 0x7FFFFFF;
      if ( v34 == *(_DWORD *)(v25 + 72) )
      {
        v45 = *(_QWORD *)(*(_QWORD *)(v40 - 8) + 32LL * *(unsigned int *)(v40 + 72) + v26);
        sub_B48D90(v25);
        v32 = v45;
        v34 = *(_DWORD *)(v25 + 4) & 0x7FFFFFF;
      }
      v27 = (v34 + 1) & 0x7FFFFFF;
      v28 = v27 | *(_DWORD *)(v25 + 4) & 0xF8000000;
      v29 = *(_QWORD *)(v25 - 8) + 32LL * (unsigned int)(v27 - 1);
      *(_DWORD *)(v25 + 4) = v28;
      if ( *(_QWORD *)v29 )
      {
        v30 = *(_QWORD *)(v29 + 8);
        **(_QWORD **)(v29 + 16) = v30;
        if ( v30 )
          *(_QWORD *)(v30 + 16) = *(_QWORD *)(v29 + 16);
      }
      *(_QWORD *)v29 = v33;
      if ( v33 )
      {
        v31 = *(_QWORD *)(v33 + 16);
        *(_QWORD *)(v29 + 8) = v31;
        if ( v31 )
          *(_QWORD *)(v31 + 16) = v29 + 8;
        *(_QWORD *)(v29 + 16) = v33 + 16;
        *(_QWORD *)(v33 + 16) = v29;
      }
      v26 += 8;
      *(_QWORD *)(*(_QWORD *)(v25 - 8)
                + 32LL * *(unsigned int *)(v25 + 72)
                + 8LL * ((*(_DWORD *)(v25 + 4) & 0x7FFFFFFu) - 1)) = v32;
    }
    while ( 8 * v41 != v26 );
    sub_B44220((_QWORD *)v25, v40 + 24, 0);
    v46[0] = (const char *)v25;
    sub_11715E0(*(_QWORD *)(a1 + 40) + 2096LL, (__int64 *)v46);
    a2 = *(_QWORD *)(v40 + 8);
    v47 = 257;
    v12 = (__int64 **)sub_B520B0(v25, a2, (__int64)v46, 0, 0);
LABEL_20:
    if ( v48 != v50 )
      _libc_free(v48, a2);
  }
  return v12;
}
