// Function: sub_11005E0
// Address: 0x11005e0
//
unsigned __int8 *__fastcall sub_11005E0(
        _QWORD *a1,
        unsigned __int8 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 *a6)
{
  __int64 v8; // r13
  __int64 v9; // r15
  unsigned __int8 v10; // al
  __int64 v11; // rdx
  int v13; // eax
  __int64 v14; // rax
  _BYTE *v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // rcx
  int v18; // edi
  __int64 v19; // rdx
  __int64 v20; // r11
  _BYTE *v21; // rsi
  char v22; // al
  char v23; // al
  _BYTE *v24; // rdi
  void *v25; // r10
  unsigned int v26; // ebx
  __int64 v27; // r13
  __int16 *v28; // rdx
  __int64 v29; // rdx
  __int64 *v30; // rdi
  int v31; // esi
  __int64 v32; // r12
  __int64 *v33; // rax
  __int64 v34; // rax
  char v35; // [rsp+8h] [rbp-108h]
  __int64 *v36; // [rsp+8h] [rbp-108h]
  __int64 *v37; // [rsp+10h] [rbp-100h]
  __int64 *v38; // [rsp+10h] [rbp-100h]
  void *v39; // [rsp+10h] [rbp-100h]
  __int64 *v40; // [rsp+10h] [rbp-100h]
  __int64 *v41; // [rsp+18h] [rbp-F8h]
  __int64 v42; // [rsp+18h] [rbp-F8h]
  __int64 v43; // [rsp+18h] [rbp-F8h]
  __int64 v44; // [rsp+18h] [rbp-F8h]
  __int64 v45; // [rsp+18h] [rbp-F8h]
  __int64 *v46; // [rsp+18h] [rbp-F8h]
  __int64 v47; // [rsp+18h] [rbp-F8h]
  _QWORD v48[2]; // [rsp+20h] [rbp-F0h] BYREF
  _BYTE *v49; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v50; // [rsp+38h] [rbp-D8h]
  _BYTE v51[64]; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v52; // [rsp+80h] [rbp-90h] BYREF
  __int16 *v53; // [rsp+88h] [rbp-88h]
  __int64 v54; // [rsp+90h] [rbp-80h]
  int v55; // [rsp+98h] [rbp-78h]
  char v56; // [rsp+9Ch] [rbp-74h]
  __int16 v57; // [rsp+A0h] [rbp-70h] BYREF

  v8 = *((_QWORD *)a2 - 4);
  v9 = *((_QWORD *)a2 + 1);
  v10 = *(_BYTE *)v8;
  if ( *(_BYTE *)v8 <= 0x15u )
  {
    v11 = sub_96F480((unsigned int)*a2 - 29, *((_QWORD *)a2 - 4), *((_QWORD *)a2 + 1), a1[11]);
    if ( v11 )
      return sub_F162A0((__int64)a1, (__int64)a2, v11);
    v10 = *(_BYTE *)v8;
  }
  if ( (unsigned __int8)(v10 - 67) <= 0xCu )
  {
    v13 = sub_10FFC90((__int64)a1, (unsigned __int8 *)v8, a2);
    if ( v13 )
    {
      v57 = 257;
      a6 = (__int64 *)sub_B51D30(v13, *(_QWORD *)(v8 - 32), v9, (__int64)&v52, 0, 0);
      v34 = *(_QWORD *)(v8 + 16);
      if ( !v34 || *(_QWORD *)(v34 + 8) )
        return (unsigned __int8 *)a6;
      goto LABEL_29;
    }
    v10 = *(_BYTE *)v8;
  }
  if ( v10 != 86 )
    goto LABEL_9;
  v15 = *(_BYTE **)(v8 - 96);
  if ( (unsigned __int8)(*v15 - 82) > 1u || *(_QWORD *)(*((_QWORD *)v15 - 8) + 8LL) != *(_QWORD *)(v8 + 8) )
    goto LABEL_57;
  if ( *a2 != 67 )
  {
LABEL_10:
    v14 = *(_QWORD *)(v8 + 16);
    if ( !v14 )
      return 0;
    a6 = *(__int64 **)(v14 + 8);
    if ( a6 )
      return 0;
    if ( *(_BYTE *)v8 != 92 )
      return 0;
    v20 = *(_QWORD *)(v8 - 64);
    if ( !v20 )
      return 0;
    v21 = *(_BYTE **)(v8 - 32);
    if ( (unsigned __int8)(*v21 - 12) <= 1u )
      goto LABEL_44;
    v42 = *(_QWORD *)(v8 - 64);
    if ( (unsigned __int8)(*v21 - 9) > 2u )
      return 0;
    v52 = 0;
    v49 = v51;
    v53 = &v57;
    v50 = 0x800000000LL;
    v54 = 8;
    v55 = 0;
    v56 = 1;
    v48[0] = &v52;
    v48[1] = &v49;
    v22 = sub_AA8FD0(v48, (__int64)v21);
    v20 = v42;
    a6 = 0;
    v35 = v22;
    if ( v22 )
    {
      while ( 1 )
      {
        v24 = v49;
        if ( !(_DWORD)v50 )
          break;
        v37 = a6;
        v21 = *(_BYTE **)&v49[8 * (unsigned int)v50 - 8];
        v43 = v20;
        LODWORD(v50) = v50 - 1;
        v23 = sub_AA8FD0(v48, (__int64)v21);
        v20 = v43;
        a6 = v37;
        if ( !v23 )
          goto LABEL_54;
      }
    }
    else
    {
LABEL_54:
      v35 = 0;
      v24 = v49;
    }
    if ( v24 != v51 )
    {
      v38 = a6;
      v44 = v20;
      _libc_free(v24, v21);
      a6 = v38;
      v20 = v44;
    }
    if ( !v56 )
    {
      v40 = a6;
      v47 = v20;
      _libc_free(v53, v21);
      a6 = v40;
      v20 = v47;
    }
    if ( !v35 )
      return 0;
LABEL_44:
    v25 = *(void **)(v8 + 72);
    v26 = *(_DWORD *)(v8 + 80);
    v27 = *(_QWORD *)(v20 + 8);
    if ( *(_BYTE *)(v27 + 8) == 17 && *(_BYTE *)(v9 + 8) == 17 && *(_DWORD *)(v9 + 32) == *(_DWORD *)(v27 + 32) )
    {
      v36 = a6;
      v39 = v25;
      v45 = v20;
      v52 = sub_BCAE30(v9);
      v53 = v28;
      v49 = (_BYTE *)sub_BCAE30(v27);
      v50 = v29;
      a6 = v36;
      if ( v49 == (_BYTE *)v52 && (_BYTE)v50 == (_BYTE)v53 )
      {
        v30 = (__int64 *)a1[4];
        v31 = *a2;
        v57 = 257;
        v32 = sub_10FF770(v30, v31 - 29, v45, v9, (__int64)&v52, 0, (int)v49, 0);
        v57 = 257;
        v33 = sub_BD2C40(112, unk_3F1FE60);
        a6 = v33;
        if ( v33 )
        {
          v46 = v33;
          sub_B4EB40((__int64)v33, v32, v39, v26, (__int64)&v52, (__int64)v33, 0);
          return (unsigned __int8 *)v46;
        }
      }
    }
    return (unsigned __int8 *)a6;
  }
  if ( (unsigned __int8)sub_F0C890((__int64)a1, *(_QWORD *)(*((_QWORD *)a2 - 4) + 8LL), *((_QWORD *)a2 + 1)) )
  {
LABEL_57:
    if ( *a2 != 78
      || (v16 = *((_QWORD *)a2 + 1),
          v17 = *(_QWORD *)(*((_QWORD *)a2 - 4) + 8LL),
          a5 = *(unsigned __int8 *)(v16 + 8),
          v18 = *(unsigned __int8 *)(v17 + 8),
          LOBYTE(a6) = (unsigned int)(v18 - 17) <= 1,
          (_BYTE)a6 == (unsigned int)(a5 - 17) <= 1)
      && ((unsigned int)(v18 - 17) > 1
       || *(_DWORD *)(v16 + 32) == *(_DWORD *)(v17 + 32) && ((_BYTE)a5 == 18) == ((_BYTE)v18 == 18)) )
    {
      a6 = (__int64 *)sub_F26350((__int64)a1, a2, v8, 0);
      if ( a6 )
      {
LABEL_29:
        v41 = a6;
        sub_F55740(v8, (__int64)a6, (__int64)a2, a1[10]);
        return (unsigned __int8 *)v41;
      }
    }
  }
  v10 = *(_BYTE *)v8;
LABEL_9:
  if ( v10 != 84 )
    goto LABEL_10;
  if ( *(_BYTE *)(*(_QWORD *)(v8 + 8) + 8LL) == 12 )
  {
    v19 = *((_QWORD *)a2 + 1);
    if ( *(_BYTE *)(v19 + 8) == 12
      && !(unsigned __int8)sub_F0C890((__int64)a1, *(_QWORD *)(*((_QWORD *)a2 - 4) + 8LL), v19) )
    {
      goto LABEL_10;
    }
  }
  a6 = (__int64 *)sub_F27020((__int64)a1, (__int64)a2, v8, 0, a5, a6);
  if ( !a6 )
    goto LABEL_10;
  return (unsigned __int8 *)a6;
}
