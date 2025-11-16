// Function: sub_1ACEBA0
// Address: 0x1aceba0
//
char __fastcall sub_1ACEBA0(__int64 a1, __int64 a2)
{
  char v3; // al
  char v4; // r13
  int v5; // eax
  char v6; // cl
  char v7; // dl
  int v8; // eax
  int v9; // eax
  char v10; // dl
  __int64 v11; // r15
  size_t v12; // r14
  unsigned __int64 v13; // r14
  _QWORD *v14; // rax
  _QWORD *v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // rdx
  char v18; // cl
  char v19; // al
  char v20; // cl
  bool v21; // zf
  char v22; // al
  char v23; // dl
  int *v25; // [rsp+8h] [rbp-108h]
  unsigned __int64 v26; // [rsp+10h] [rbp-100h] BYREF
  __int64 v27[2]; // [rsp+20h] [rbp-F0h] BYREF
  __int64 v28; // [rsp+30h] [rbp-E0h] BYREF
  int v29[4]; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v30; // [rsp+50h] [rbp-C0h] BYREF

  if ( (*(_BYTE *)(a2 + 23) & 0x20) != 0 )
  {
    v11 = *(_QWORD *)(a1 + 8);
    sub_15E4EB0(v27, a2);
    v12 = v27[1];
    v25 = (int *)v27[0];
    sub_16C1840(v29);
    sub_16C1A90(v29, v25, v12);
    sub_16C1AA0(v29, &v26);
    v13 = v26;
    if ( (__int64 *)v27[0] != &v28 )
      j_j___libc_free_0(v27[0], v28 + 1);
    v14 = *(_QWORD **)(v11 + 16);
    v15 = (_QWORD *)(v11 + 8);
    if ( v14 )
    {
      do
      {
        while ( 1 )
        {
          v16 = v14[2];
          v17 = v14[3];
          if ( v13 <= v14[4] )
            break;
          v14 = (_QWORD *)v14[3];
          if ( !v17 )
            goto LABEL_36;
        }
        v15 = v14;
        v14 = (_QWORD *)v14[2];
      }
      while ( v16 );
LABEL_36:
      if ( (_QWORD *)(v11 + 8) != v15 && v13 >= v15[4] )
      {
        *(_QWORD *)v29 = (4LL * *(unsigned __int8 *)(v11 + 178)) | (unsigned __int64)(v15 + 4) & 0xFFFFFFFFFFFFFFFBLL;
        if ( (*(_QWORD *)v29 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          if ( (unsigned __int8)sub_16341C0(v29) )
          {
            v18 = *(_BYTE *)(a2 + 33);
            v19 = v18 & 3;
            v20 = v18 | 0x40;
            v21 = v19 == 1;
            v22 = v20;
            if ( v21 )
              v22 = v20 & 0xFC;
            *(_BYTE *)(a2 + 33) = v22;
          }
        }
      }
    }
  }
  if ( (*(_BYTE *)(a2 + 32) & 0xFu) - 7 <= 1 )
  {
    v3 = sub_1ACE610(a1, a2);
    if ( v3 || *(_QWORD *)(a1 + 16) )
    {
      v4 = v3;
      sub_1ACE9A0((__int64)v29, a1, a2, v3);
      LOWORD(v28) = 260;
      v27[0] = (__int64)v29;
      sub_164B780(a2, v27);
      if ( *(__int64 **)v29 != &v30 )
        j_j___libc_free_0(*(_QWORD *)v29, v30 + 1);
      v5 = sub_1ACEAA0(a1, a2, v4);
      v6 = v5 & 0xF;
      v7 = *(_BYTE *)(a2 + 32);
      if ( (unsigned int)(v5 - 7) > 1 )
      {
        v5 &= 0xFu;
        v23 = v6 | v7 & 0xF0;
        *(_BYTE *)(a2 + 32) = v23;
        if ( (unsigned int)(v5 - 7) > 1 && ((v23 & 0x30) == 0 || v6 == 9) )
          goto LABEL_10;
      }
      else
      {
        *(_BYTE *)(a2 + 32) = v6 | v7 & 0xC0;
      }
      *(_BYTE *)(a2 + 33) |= 0x40u;
      if ( v5 == 7 )
        goto LABEL_12;
LABEL_10:
      if ( v5 == 8 || (*(_BYTE *)(a2 + 32) = *(_BYTE *)(a2 + 32) & 0xCF | 0x10, v6 == 9) )
      {
LABEL_12:
        v8 = *(unsigned __int8 *)(a2 + 16);
        if ( *(_BYTE *)(a2 + 16) && v8 != 3 )
          return v8;
LABEL_14:
        LOBYTE(v8) = sub_15E4F60(a2);
        if ( !(_BYTE)v8 )
          return v8;
        goto LABEL_22;
      }
      goto LABEL_18;
    }
  }
  v9 = sub_1ACEAA0(a1, a2, 0);
  if ( (unsigned int)(v9 - 7) <= 1 )
  {
    *(_BYTE *)(a2 + 32) = *(_BYTE *)(a2 + 32) & 0xC0 | v9 & 0xF;
LABEL_18:
    *(_BYTE *)(a2 + 33) |= 0x40u;
    goto LABEL_19;
  }
  v10 = v9 & 0xF | *(_BYTE *)(a2 + 32) & 0xF0;
  *(_BYTE *)(a2 + 32) = v10;
  if ( (v9 & 0xFu) - 7 <= 1 || (v10 & 0x30) != 0 && (v9 & 0xF) != 9 )
    goto LABEL_18;
LABEL_19:
  v8 = *(unsigned __int8 *)(a2 + 16);
  if ( v8 != 3 && *(_BYTE *)(a2 + 16) )
    return v8;
  LOBYTE(v8) = *(_BYTE *)(a2 + 32) & 0xF;
  if ( (_BYTE)v8 != 1 )
    goto LABEL_14;
LABEL_22:
  if ( *(_QWORD *)(a2 + 48) )
    *(_QWORD *)(a2 + 48) = 0;
  return v8;
}
