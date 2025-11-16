// Function: sub_DFF9C0
// Address: 0xdff9c0
//
__int64 __fastcall sub_DFF9C0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, _BYTE *a5)
{
  unsigned int v5; // r12d
  __int64 v6; // r9
  __int64 v7; // r11
  bool v8; // r10
  __int64 v9; // rax
  _BYTE *v10; // rdx
  unsigned __int8 *v11; // r15
  __int64 v12; // rsi
  _BYTE *v13; // rdi
  __int64 v14; // rax
  _QWORD *v15; // r13
  bool v16; // si
  unsigned __int8 **v17; // r8
  unsigned __int8 **v18; // rdx
  unsigned __int8 *v19; // rax
  int v20; // edi
  unsigned int v21; // ecx
  __int64 v22; // rax
  unsigned __int8 *v23; // rdx
  bool v24; // dl
  unsigned __int8 v25; // al
  unsigned __int8 *v26; // r14
  unsigned int v27; // edi
  __int64 v29; // rax
  _QWORD *v30; // rdx
  int v31; // r15d
  unsigned int v32; // eax
  __int64 v33; // rcx
  _QWORD *v34; // rdx
  unsigned int v35; // eax
  __int64 v36; // rcx
  _QWORD *v37; // rdx
  unsigned int v38; // eax
  unsigned int v39; // r14d
  __int64 v40; // rsi
  _QWORD *v41; // rcx
  __int64 v42; // rcx
  unsigned __int8 *v43; // rcx
  unsigned __int8 *v44; // rcx
  _BYTE **v45; // rax
  _BYTE *v46; // rsi
  __int64 v47; // [rsp+0h] [rbp-60h]
  unsigned __int8 v51; // [rsp+26h] [rbp-3Ah]
  unsigned __int8 v52; // [rsp+27h] [rbp-39h]

  v6 = a1;
  v7 = a2;
  v47 = a2;
  v52 = *(_BYTE *)(a1 - 16);
  v8 = (v52 & 2) != 0;
  if ( (v52 & 2) != 0 )
  {
    v9 = *(_QWORD *)(a1 - 32);
    v10 = *(_BYTE **)(v9 + 8);
    if ( !v10 )
      goto LABEL_5;
  }
  else
  {
    v9 = a1 - 8LL * ((v52 >> 2) & 0xF) - 16;
    v10 = *(_BYTE **)(v9 + 8);
    if ( !v10 )
      goto LABEL_5;
  }
  if ( (unsigned __int8)(*v10 - 5) >= 0x20u )
    v10 = 0;
LABEL_5:
  v11 = *(unsigned __int8 **)v9;
  if ( *(_QWORD *)v9 && (unsigned __int8)(*v11 - 5) >= 0x20u )
    v11 = 0;
  LOBYTE(v5) = v11 == v10 && a3 == (_QWORD)v10;
  if ( (_BYTE)v5 )
  {
    if ( a4 )
      *a4 = sub_DFF6F0(a3);
    goto LABEL_90;
  }
  if ( (v52 & 2) != 0 )
  {
    v12 = *(_QWORD *)(a1 - 32);
    if ( *(_DWORD *)(a1 - 24) <= 3u )
      goto LABEL_14;
    v13 = *(_BYTE **)(v12 + 8);
    if ( !v13 || (unsigned __int8)(*v13 - 5) > 0x1Fu )
    {
      v5 = (v52 & 2) != 0;
      goto LABEL_14;
    }
LABEL_60:
    LOBYTE(v38) = sub_DFF670((__int64)v13);
    v5 = v38;
    goto LABEL_14;
  }
  v12 = a1 - 8LL * ((v52 >> 2) & 0xF) - 16;
  if ( ((*(_WORD *)(a1 - 16) >> 6) & 0xFu) <= 3 )
    goto LABEL_14;
  v13 = *(_BYTE **)(v12 + 8);
  if ( v13 )
  {
    v5 = 1;
    if ( (unsigned __int8)(*v13 - 5) > 0x1Fu )
      goto LABEL_14;
    goto LABEL_60;
  }
  v5 = 1;
LABEL_14:
  v14 = *(_QWORD *)(*(_QWORD *)(v12 + 16) + 136LL);
  v15 = *(_QWORD **)(v14 + 24);
  if ( *(_DWORD *)(v14 + 32) > 0x40u )
    v15 = (_QWORD *)*v15;
  if ( v11 )
  {
    v51 = *(_BYTE *)(v7 - 16);
    v16 = (v51 & 2) != 0;
    v17 = (unsigned __int8 **)(v7 + -16 - 8LL * ((v51 >> 2) & 0xF));
    while ( 1 )
    {
      v18 = v17;
      if ( (v51 & 2) != 0 )
        v18 = *(unsigned __int8 ***)(v7 - 32);
      v19 = *v18;
      if ( *v18 )
      {
        v20 = *v19;
        v21 = v20 - 5;
        LOBYTE(v21) = v11 == v19 && (unsigned __int8)(v20 - 5) <= 0x1Fu;
        if ( (_BYTE)v21 )
          break;
      }
      if ( (_BYTE)v5 )
      {
        v22 = v8 ? *(_QWORD *)(v6 - 32) : v6 + -16 - 8LL * ((v52 >> 2) & 0xF);
        v23 = *(unsigned __int8 **)(v22 + 8);
        if ( v23 )
        {
          if ( (unsigned __int8)(*v23 - 5) <= 0x1Fu && v11 == v23 )
            goto LABEL_82;
        }
      }
      v24 = sub_DFF670((__int64)v11);
      v25 = *(v11 - 16);
      if ( (v25 & 2) != 0 )
      {
        v26 = (unsigned __int8 *)*((_QWORD *)v11 - 4);
        v27 = *((_DWORD *)v11 - 6);
      }
      else
      {
        v27 = (*((_WORD *)v11 - 8) >> 6) & 0xF;
        v26 = &v11[-16 - 8LL * ((v25 >> 2) & 0xF)];
      }
      if ( v24 )
      {
        if ( v27 > 5 )
        {
          v31 = 3;
          v32 = 3;
          goto LABEL_46;
        }
        goto LABEL_32;
      }
      if ( v27 <= 1 )
        goto LABEL_32;
      if ( v27 > 3 )
      {
        v31 = 2;
        v32 = 1;
        while ( 1 )
        {
LABEL_46:
          v33 = *(_QWORD *)(*(_QWORD *)&v26[8 * v32 + 8] + 136LL);
          v34 = *(_QWORD **)(v33 + 24);
          if ( *(_DWORD *)(v33 + 32) > 0x40u )
            v34 = (_QWORD *)*v34;
          if ( v34 > v15 )
            break;
          v32 += v31;
          if ( v32 >= v27 )
            goto LABEL_55;
        }
        v35 = v32 - v31;
        if ( v35 )
          goto LABEL_50;
LABEL_55:
        v35 = v27 - v31;
LABEL_50:
        v36 = *(_QWORD *)(*(_QWORD *)&v26[8 * v35 + 8] + 136LL);
        v37 = *(_QWORD **)(v36 + 24);
        if ( *(_DWORD *)(v36 + 32) > 0x40u )
          v37 = (_QWORD *)*v37;
        v11 = *(unsigned __int8 **)&v26[8 * v35];
        if ( !v11 || (unsigned __int8)(*v11 - 5) > 0x1Fu )
          goto LABEL_32;
        v15 = (_QWORD *)((char *)v15 - (__int64)v37);
      }
      else
      {
        if ( v27 != 2 )
        {
          v29 = *(_QWORD *)(*((_QWORD *)v26 + 2) + 136LL);
          v30 = *(_QWORD **)(v29 + 24);
          if ( *(_DWORD *)(v29 + 32) <= 0x40u )
            v15 = (_QWORD *)((char *)v15 - (__int64)v30);
          else
            v15 = (_QWORD *)((char *)v15 - *v30);
        }
        v11 = (unsigned __int8 *)*((_QWORD *)v26 + 1);
        if ( !v11 || (unsigned __int8)(*v11 - 5) > 0x1Fu )
          goto LABEL_32;
      }
    }
    v39 = v21;
    v40 = *((_QWORD *)v18[2] + 17);
    v41 = *(_QWORD **)(v40 + 24);
    if ( *(_DWORD *)(v40 + 32) > 0x40u )
      v41 = (_QWORD *)*v41;
    if ( v41 == v15
      || (v8 ? (v42 = *(_QWORD *)(v6 - 32)) : (v42 = v6 - 8LL * ((v52 >> 2) & 0xF) - 16),
          (v43 = *(unsigned __int8 **)(v42 + 8)) != 0 && (unsigned __int8)(*v43 - 5) <= 0x1Fu && v11 == v43
       || (v44 = v18[1]) != 0 && (unsigned __int8)(*v44 - 5) <= 0x1Fu && v19 == v44) )
    {
      *a5 = 1;
      if ( !a4 )
        return v39;
    }
    else
    {
      *a5 = 0;
      if ( !a4 )
        return v39;
      v47 = sub_DFF6F0(a3);
    }
    *a4 = v47;
    return v39;
  }
LABEL_32:
  if ( (_BYTE)v5 )
  {
    v11 = 0;
    v51 = *(_BYTE *)(v7 - 16);
    v16 = (v51 & 2) != 0;
LABEL_82:
    if ( v16 )
      v45 = *(_BYTE ***)(v7 - 32);
    else
      v45 = (_BYTE **)(v7 - 8LL * ((v51 >> 2) & 0xF) - 16);
    v46 = *v45;
    if ( *v45 && (unsigned __int8)(*v46 - 5) >= 0x20u )
      v46 = 0;
    v5 = sub_DFF840((__int64)v11, v46);
    if ( (_BYTE)v5 )
    {
      if ( a4 )
        *a4 = sub_DFF6F0(a3);
LABEL_90:
      *a5 = 1;
    }
  }
  return v5;
}
