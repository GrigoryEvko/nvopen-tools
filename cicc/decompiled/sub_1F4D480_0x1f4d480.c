// Function: sub_1F4D480
// Address: 0x1f4d480
//
__int64 __fastcall sub_1F4D480(__int64 a1, int a2, int a3, __int64 a4, unsigned __int64 a5, unsigned int a6)
{
  unsigned int v6; // r15d
  int v11; // r12d
  unsigned int v12; // eax
  int v13; // eax
  int v14; // eax
  char v15; // si
  __int64 v16; // r9
  __int64 v17; // rsi
  int v18; // edi
  __int64 v19; // rax
  int v20; // r10d
  int v21; // r11d
  int v22; // edi
  __int64 v23; // rax
  int v24; // r10d
  int v25; // r11d
  __int64 v26; // r9
  __int64 v27; // r10
  __int64 v28; // r8
  _WORD *v29; // r11
  unsigned __int16 v30; // di
  __int16 *v31; // r9
  _WORD *v32; // r11
  unsigned __int16 *v33; // r8
  unsigned int v34; // edx
  unsigned int v35; // r10d
  unsigned int v36; // r11d
  __int16 *v37; // r10
  __int16 v38; // r9
  __int64 v39; // rdi
  __int64 v40; // r9
  __int64 v41; // r10
  _WORD *v42; // r11
  unsigned __int16 v43; // si
  __int16 *v44; // rdi
  _WORD *v45; // r11
  unsigned __int16 *v46; // r10
  unsigned int v47; // r11d
  unsigned int v48; // r8d
  unsigned int v49; // r9d
  __int16 *v50; // r11
  __int16 v51; // di
  int v52; // r9d
  int v53; // edx
  int v54; // [rsp+Ch] [rbp-54h]
  int v55; // [rsp+18h] [rbp-48h]
  unsigned int v57; // [rsp+28h] [rbp-38h] BYREF
  _DWORD v58[13]; // [rsp+2Ch] [rbp-34h] BYREF

  if ( !*(_DWORD *)(a1 + 296) )
    return 0;
  v11 = a4;
  LOBYTE(v12) = sub_1F4D060(a5, a4, *(_QWORD *)(a1 + 280), a4, a5, a6);
  v6 = v12;
  if ( !(_BYTE)v12 )
    return 0;
  v13 = sub_1F4CA20(a2, a1 + 584);
  if ( !v13 )
    goto LABEL_11;
  v54 = v13;
  v55 = sub_1F4CA20(a3, a1 + 552);
  v14 = sub_1F4CA20(v11, a1 + 552);
  v15 = 0;
  if ( v55 )
  {
    v15 = v6;
    if ( v54 != v55 )
    {
      if ( v55 < 0 || v54 < 0 )
      {
LABEL_47:
        v15 = 0;
      }
      else
      {
        v39 = *(_QWORD *)(a1 + 248);
        v40 = *(_QWORD *)(v39 + 8);
        v41 = *(_QWORD *)(v39 + 56);
        LODWORD(v39) = *(_DWORD *)(v40 + 24LL * (unsigned int)v55 + 16);
        v42 = (_WORD *)(v41 + 2LL * ((unsigned int)v39 >> 4));
        v43 = *v42 + v55 * (v39 & 0xF);
        v44 = v42 + 1;
        LODWORD(v42) = *(_DWORD *)(v40 + 24LL * (unsigned int)v54 + 16);
        LODWORD(v40) = v54 * ((unsigned __int8)v42 & 0xF);
        v45 = (_WORD *)(v41 + 2LL * ((unsigned int)v42 >> 4));
        LOWORD(v49) = *v45 + v40;
        v46 = v45 + 1;
        v47 = v43;
        v48 = v49;
        v49 = (unsigned __int16)v49;
        while ( v47 != v49 )
        {
          if ( v47 >= v49 )
          {
            v52 = *v46;
            if ( !(_WORD)v52 )
              goto LABEL_47;
            v48 += v52;
            ++v46;
            v49 = (unsigned __int16)v48;
          }
          else
          {
            v50 = v44 + 1;
            v51 = *v44;
            v43 += v51;
            if ( !v51 )
              goto LABEL_47;
            v44 = v50;
            v47 = v43;
          }
        }
        v15 = v6;
      }
    }
  }
  if ( !v14 )
  {
    if ( v55 && v15 != 1 )
      return v6;
    if ( !v15 )
      goto LABEL_11;
    return 0;
  }
  if ( v54 != v14 )
  {
    if ( v14 < 0 || v54 < 0 )
    {
LABEL_34:
      if ( !v15 && v55 )
        goto LABEL_11;
      return 0;
    }
    v26 = *(_QWORD *)(a1 + 248);
    v27 = *(_QWORD *)(v26 + 8);
    v28 = *(_QWORD *)(v26 + 56);
    LODWORD(v26) = *(_DWORD *)(v27 + 24LL * (unsigned int)v14 + 16);
    v29 = (_WORD *)(v28 + 2LL * ((unsigned int)v26 >> 4));
    v30 = *v29 + v14 * (v26 & 0xF);
    v31 = v29 + 1;
    LODWORD(v29) = *(_DWORD *)(v27 + 24LL * (unsigned int)v54 + 16);
    LOWORD(v27) = v54 * ((unsigned __int8)v29 & 0xF);
    v32 = (_WORD *)(v28 + 2LL * ((unsigned int)v29 >> 4));
    v33 = v32 + 1;
    v34 = (unsigned __int16)(*v32 + v27);
    v35 = v30;
    v36 = v34;
    while ( v35 != v34 )
    {
      if ( v35 >= v34 )
      {
        v53 = *v33;
        if ( !(_WORD)v53 )
          goto LABEL_34;
        v36 += v53;
        ++v33;
        v34 = (unsigned __int16)v36;
      }
      else
      {
        v37 = v31 + 1;
        v38 = *v31;
        v30 += v38;
        if ( !v38 )
          goto LABEL_34;
        v31 = v37;
        v35 = v30;
      }
    }
  }
  if ( !v55 || !v15 )
    return v6;
LABEL_11:
  v57 = 0;
  if ( !(unsigned __int8)sub_1F4CAA0(a1, v11, a6, &v57) )
    return 0;
  v58[0] = 0;
  if ( (unsigned __int8)sub_1F4CAA0(a1, a3, a6, v58) )
  {
    if ( dword_4FCE300 > 0 )
    {
      v16 = *(_QWORD *)(a1 + 264);
      v17 = *(_QWORD *)(a1 + 304);
      v18 = v11;
      do
      {
        v19 = sub_1F4C7B0(v18, v17, v16);
        if ( !v19 || **(_WORD **)(v19 + 16) != 15 )
          break;
        v18 = *(_DWORD *)(*(_QWORD *)(v19 + 32) + 48LL);
        if ( a2 == v18 )
          return v6;
      }
      while ( v20 != v21 + 1 );
      v22 = a3;
      do
      {
        v23 = sub_1F4C7B0(v22, v17, v16);
        if ( !v23 || **(_WORD **)(v23 + 16) != 15 )
          break;
        v22 = *(_DWORD *)(*(_QWORD *)(v23 + 32) + 48LL);
        if ( a2 == v22 )
          return 0;
      }
      while ( v24 != v25 + 1 );
    }
    if ( !v58[0] || v58[0] >= v57 )
      return 0;
  }
  return v6;
}
