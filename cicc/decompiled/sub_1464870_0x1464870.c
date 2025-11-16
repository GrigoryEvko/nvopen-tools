// Function: sub_1464870
// Address: 0x1464870
//
void __fastcall sub_1464870(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // eax
  unsigned __int64 v5; // rdi
  _BYTE *v6; // r8
  char v7; // dl
  int v8; // ebx
  int v9; // eax
  char v10; // bl
  unsigned int v11; // edx
  __int64 v12; // rcx
  __int64 v13; // rdi
  __int64 v14; // rbx
  char v15; // al
  __int64 v16; // r12
  __int64 *v17; // rsi
  __int64 *v18; // rax
  __int64 *v19; // rcx
  __int64 v20; // r11
  int v21; // r9d
  __int64 v22; // r10
  __int64 v24; // [rsp+20h] [rbp-1A0h]
  __int64 v25; // [rsp+20h] [rbp-1A0h]
  __int64 v26; // [rsp+20h] [rbp-1A0h]
  __int64 v27; // [rsp+20h] [rbp-1A0h]
  void *v29; // [rsp+30h] [rbp-190h] BYREF
  char v30[16]; // [rsp+38h] [rbp-188h] BYREF
  __int64 v31; // [rsp+48h] [rbp-178h]
  void *v32; // [rsp+60h] [rbp-160h] BYREF
  char v33[16]; // [rsp+68h] [rbp-158h] BYREF
  __int64 v34; // [rsp+78h] [rbp-148h]
  __int64 v35; // [rsp+90h] [rbp-130h] BYREF
  _BYTE *v36; // [rsp+98h] [rbp-128h]
  _BYTE *v37; // [rsp+A0h] [rbp-120h]
  __int64 v38; // [rsp+A8h] [rbp-118h]
  int v39; // [rsp+B0h] [rbp-110h]
  _BYTE v40[72]; // [rsp+B8h] [rbp-108h] BYREF
  _BYTE *v41; // [rsp+100h] [rbp-C0h] BYREF
  __int64 v42; // [rsp+108h] [rbp-B8h]
  _BYTE v43[176]; // [rsp+110h] [rbp-B0h] BYREF

  v41 = v43;
  v42 = 0x1000000000LL;
  sub_1453C10(a2, (__int64)&v41);
  v36 = v40;
  v35 = 0;
  v37 = v40;
  v38 = 8;
  v39 = 0;
  sub_1412190((__int64)&v35, a2);
  v4 = v42;
  v5 = (unsigned __int64)v37;
  v6 = v36;
  if ( (_DWORD)v42 )
  {
    while ( 1 )
    {
      v16 = *(_QWORD *)&v41[8 * v4 - 8];
      LODWORD(v42) = v4 - 1;
      if ( (_BYTE *)v5 == v6 )
      {
        v17 = (__int64 *)(v5 + 8LL * HIDWORD(v38));
        if ( v17 != (__int64 *)v5 )
        {
          v18 = (__int64 *)v5;
          v19 = 0;
          while ( v16 != *v18 )
          {
            if ( *v18 == -2 )
              v19 = v18;
            if ( v17 == ++v18 )
            {
              if ( !v19 )
                goto LABEL_36;
              *v19 = v16;
              --v39;
              ++v35;
              goto LABEL_4;
            }
          }
          goto LABEL_21;
        }
LABEL_36:
        if ( HIDWORD(v38) < (unsigned int)v38 )
          break;
      }
      sub_16CCBA0(&v35, v16);
      v5 = (unsigned __int64)v37;
      v6 = v36;
      if ( v7 )
        goto LABEL_4;
LABEL_21:
      v4 = v42;
      if ( !(_DWORD)v42 )
        goto LABEL_31;
    }
    ++HIDWORD(v38);
    *v17 = v16;
    ++v35;
LABEL_4:
    v8 = *(_DWORD *)(a1 + 168);
    if ( v8 )
    {
      v24 = *(_QWORD *)(a1 + 152);
      sub_1457D90(&v29, -8, 0);
      sub_1457D90(&v32, -16, 0);
      v9 = v8 - 1;
      v10 = 1;
      v11 = v9 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v12 = v24 + 48LL * v11;
      v13 = *(_QWORD *)(v12 + 24);
      if ( v16 != v13 )
      {
        v20 = v24 + 48LL * (v9 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4)));
        v21 = 1;
        v12 = 0;
        while ( v13 != v31 )
        {
          if ( v12 || v13 != v34 )
            v20 = v12;
          v11 = v9 & (v21 + v11);
          v12 = v24 + 48LL * v11;
          v13 = *(_QWORD *)(v12 + 24);
          if ( v16 == v13 )
          {
            v10 = 1;
            goto LABEL_6;
          }
          ++v21;
          v22 = v20;
          v20 = v24 + 48LL * v11;
          v12 = v22;
        }
        v10 = 0;
        if ( !v12 )
          v12 = v20;
      }
LABEL_6:
      v32 = &unk_49EE2B0;
      if ( v34 != 0 && v34 != -8 && v34 != -16 )
      {
        v25 = v12;
        sub_1649B30(v33);
        v12 = v25;
      }
      v29 = &unk_49EE2B0;
      if ( v31 != -8 && v31 != 0 && v31 != -16 )
      {
        v26 = v12;
        sub_1649B30(v30);
        v12 = v26;
      }
      if ( v10 && v12 != *(_QWORD *)(a1 + 152) + 48LL * *(unsigned int *)(a1 + 168) )
      {
        v14 = *(_QWORD *)(v12 + 40);
        if ( v14 != a3 )
        {
          v27 = v12;
          v15 = sub_14594A0(a1, *(_QWORD *)(v12 + 40), a3);
          v12 = v27;
          if ( !v15 )
            goto LABEL_20;
        }
        if ( *(_BYTE *)(v16 + 16) != 77 || *(_WORD *)(v14 + 24) != 10 || v14 == a3 && a2 != v16 )
        {
          sub_1464220(a1, *(_QWORD *)(v12 + 24));
          sub_1459590(a1, v14);
        }
      }
    }
    sub_1453C10(v16, (__int64)&v41);
LABEL_20:
    v5 = (unsigned __int64)v37;
    v6 = v36;
    goto LABEL_21;
  }
LABEL_31:
  if ( v6 != (_BYTE *)v5 )
    _libc_free(v5);
  if ( v41 != v43 )
    _libc_free((unsigned __int64)v41);
}
