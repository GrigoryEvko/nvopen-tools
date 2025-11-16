// Function: sub_D9BB00
// Address: 0xd9bb00
//
__int64 __fastcall sub_D9BB00(__int64 a1, __int64 a2, unsigned __int8 *a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  unsigned int v10; // r12d
  __int64 v13; // rsi
  __int64 *v14; // r8
  __int64 v15; // r9
  __int64 v16; // rcx
  __int64 i; // rax
  __int64 *v18; // rdx
  __int64 v19; // r15
  __int64 *v20; // rax
  char v21; // dl
  char *v22; // rax
  char *v23; // rdx
  unsigned __int8 v24; // al
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // r15
  __int64 v30; // rax
  __int64 v31; // [rsp+0h] [rbp-150h]
  __int64 *v32; // [rsp+8h] [rbp-148h]
  __int64 *v33; // [rsp+20h] [rbp-130h] BYREF
  __int64 v34; // [rsp+28h] [rbp-128h]
  _QWORD v35[6]; // [rsp+30h] [rbp-120h] BYREF
  __int64 v36; // [rsp+60h] [rbp-F0h] BYREF
  char *v37; // [rsp+68h] [rbp-E8h]
  __int64 v38; // [rsp+70h] [rbp-E0h]
  int v39; // [rsp+78h] [rbp-D8h]
  char v40; // [rsp+7Ch] [rbp-D4h]
  char v41; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v42; // [rsp+C0h] [rbp-90h] BYREF
  __int64 *v43; // [rsp+C8h] [rbp-88h]
  __int64 v44; // [rsp+D0h] [rbp-80h]
  int v45; // [rsp+D8h] [rbp-78h]
  unsigned __int8 v46; // [rsp+DCh] [rbp-74h]
  char v47; // [rsp+E0h] [rbp-70h] BYREF

  v10 = sub_98F650((__int64)a3, a2, a3, a4, a5);
  if ( (_BYTE)v10 )
    return v10;
  v13 = (__int64)&v36;
  v37 = &v41;
  v36 = 0;
  v38 = 8;
  v39 = 0;
  v40 = 1;
  sub_D9B980(a1, (__int64)&v36, a2, v7, v8, v9);
  v46 = 1;
  v16 = 1;
  v33 = v35;
  v42 = 0;
  v44 = 8;
  v45 = 0;
  v43 = (__int64 *)&v47;
  v35[0] = a3;
  v34 = 0x600000001LL;
  LODWORD(i) = 1;
  while ( 2 )
  {
    if ( !(_DWORD)i )
    {
LABEL_12:
      v10 = 1;
LABEL_13:
      if ( !(_BYTE)v16 )
        goto LABEL_55;
      goto LABEL_14;
    }
    while ( 1 )
    {
      v18 = v33;
      v19 = v33[(unsigned int)i - 1];
      LODWORD(v34) = i - 1;
      if ( (_BYTE)v16 )
      {
        v20 = v43;
        v13 = HIDWORD(v44);
        v18 = &v43[HIDWORD(v44)];
        if ( v43 != v18 )
        {
          while ( v19 != *v20 )
          {
            if ( v18 == ++v20 )
              goto LABEL_26;
          }
          goto LABEL_10;
        }
LABEL_26:
        if ( HIDWORD(v44) < (unsigned int)v44 )
          break;
      }
      v13 = v19;
      sub_C8CC70((__int64)&v42, v19, (__int64)v18, v16, (__int64)v14, v15);
      v16 = v46;
      if ( v21 )
        goto LABEL_19;
LABEL_10:
      LODWORD(i) = v34;
LABEL_11:
      if ( !(_DWORD)i )
        goto LABEL_12;
    }
    v13 = (unsigned int)++HIDWORD(v44);
    *v18 = v19;
    v16 = v46;
    ++v42;
LABEL_19:
    if ( (unsigned int)(HIDWORD(v44) - v45) > 0x10 )
      goto LABEL_13;
    if ( v40 )
    {
      v22 = v37;
      v23 = &v37[8 * HIDWORD(v38)];
      if ( v37 == v23 )
        goto LABEL_28;
      while ( v19 != *(_QWORD *)v22 )
      {
        v22 += 8;
        if ( v23 == v22 )
          goto LABEL_28;
      }
      goto LABEL_10;
    }
    v13 = v19;
    if ( sub_C8CA60((__int64)&v36, v19) )
    {
      v16 = v46;
      LODWORD(i) = v34;
      goto LABEL_11;
    }
LABEL_28:
    v13 = 0;
    if ( sub_98ED70((unsigned __int8 *)v19, 0, 0, 0, 0) )
      goto LABEL_52;
    v24 = *(_BYTE *)v19;
    if ( *(_BYTE *)v19 <= 0x1Cu )
      goto LABEL_54;
    if ( v24 != 58 )
    {
      if ( v24 == 85
        && (v30 = *(_QWORD *)(v19 - 32)) != 0
        && !*(_BYTE *)v30
        && *(_QWORD *)(v30 + 24) == *(_QWORD *)(v19 + 80)
        && (*(_BYTE *)(v30 + 33) & 0x20) != 0
        && *(_DWORD *)(v30 + 36) == 493 )
      {
LABEL_52:
        LODWORD(i) = v34;
      }
      else
      {
LABEL_32:
        v13 = 0;
        if ( sub_98CD70((unsigned __int8 *)v19, 0) )
          goto LABEL_54;
        if ( (unsigned __int8)sub_B44920(v19)
          || (unsigned __int8)sub_B44AB0((unsigned __int8 *)v19)
          || (unsigned __int8)sub_B44930(v19) )
        {
          v27 = *(unsigned int *)(a4 + 8);
          if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
          {
            v13 = a4 + 16;
            sub_C8D5F0(a4, (const void *)(a4 + 16), v27 + 1, 8u, v25, v26);
            v27 = *(unsigned int *)(a4 + 8);
          }
          *(_QWORD *)(*(_QWORD *)a4 + 8 * v27) = v19;
          ++*(_DWORD *)(a4 + 8);
        }
        v28 = 4LL * (*(_DWORD *)(v19 + 4) & 0x7FFFFFF);
        if ( (*(_BYTE *)(v19 + 7) & 0x40) != 0 )
        {
          v14 = *(__int64 **)(v19 - 8);
          v15 = (__int64)&v14[v28];
        }
        else
        {
          v15 = v19;
          v14 = (__int64 *)(v19 - v28 * 8);
        }
        for ( i = (unsigned int)v34; (__int64 *)v15 != v14; LODWORD(v34) = v34 + 1 )
        {
          v29 = *v14;
          if ( i + 1 > (unsigned __int64)HIDWORD(v34) )
          {
            v13 = (__int64)v35;
            v31 = v15;
            v32 = v14;
            sub_C8D5F0((__int64)&v33, v35, i + 1, 8u, (__int64)v14, v15);
            i = (unsigned int)v34;
            v15 = v31;
            v14 = v32;
          }
          v14 += 4;
          v33[i] = v29;
          i = (unsigned int)(v34 + 1);
        }
      }
      v16 = v46;
      continue;
    }
    break;
  }
  if ( (*(_BYTE *)(v19 + 1) & 2) == 0 )
    goto LABEL_32;
LABEL_54:
  v10 = 0;
  if ( !v46 )
LABEL_55:
    _libc_free(v43, v13);
LABEL_14:
  if ( v33 != v35 )
    _libc_free(v33, v13);
  if ( !v40 )
    _libc_free(v37, v13);
  return v10;
}
