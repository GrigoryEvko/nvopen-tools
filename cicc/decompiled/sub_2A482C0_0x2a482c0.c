// Function: sub_2A482C0
// Address: 0x2a482c0
//
void __fastcall sub_2A482C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5, __int64 a6)
{
  __int64 *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rcx
  unsigned int v11; // eax
  __int64 v12; // rbx
  __int64 *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // r12
  __int64 *v16; // r12
  __int64 *v17; // r13
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // r13
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __int64 v25; // rax
  _BYTE *v26; // rdi
  __int64 *v28; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v29; // [rsp+28h] [rbp-C8h]
  _QWORD v30[4]; // [rsp+30h] [rbp-C0h] BYREF
  __int64 *v31; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v32; // [rsp+58h] [rbp-98h]
  __int64 v33; // [rsp+60h] [rbp-90h] BYREF
  char v34; // [rsp+68h] [rbp-88h] BYREF
  __int64 v35; // [rsp+80h] [rbp-70h] BYREF
  __int64 *v36; // [rsp+88h] [rbp-68h]
  __int64 v37; // [rsp+90h] [rbp-60h]
  int v38; // [rsp+98h] [rbp-58h]
  unsigned __int8 v39; // [rsp+9Ch] [rbp-54h]
  char v40; // [rsp+A0h] [rbp-50h] BYREF

  v7 = v30;
  v36 = (__int64 *)&v40;
  v8 = -(__int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v28 = v30;
  v35 = 0;
  v9 = *(_QWORD *)(a2 + 32 * v8);
  v37 = 4;
  v38 = 0;
  v39 = 1;
  v30[0] = v9;
  v10 = 1;
  v29 = 0x400000001LL;
  v11 = 1;
  while ( 1 )
  {
    v12 = v7[v11 - 1];
    LODWORD(v29) = v11 - 1;
    if ( (_BYTE)v10 )
    {
      v13 = v36;
      v7 = &v36[HIDWORD(v37)];
      if ( v36 != v7 )
      {
        while ( v12 != *v13 )
        {
          if ( v7 == ++v13 )
            goto LABEL_35;
        }
        goto LABEL_7;
      }
LABEL_35:
      if ( HIDWORD(v37) < (unsigned int)v37 )
        break;
    }
    sub_C8CC70((__int64)&v35, v12, (__int64)v7, v10, (__int64)a5, a6);
    v10 = v39;
    if ( (_BYTE)v7 )
      goto LABEL_10;
LABEL_7:
    v11 = v29;
    if ( !(_DWORD)v29 )
      goto LABEL_30;
LABEL_8:
    v7 = v28;
  }
  ++HIDWORD(v37);
  *v7 = v12;
  v10 = v39;
  ++v35;
LABEL_10:
  if ( (unsigned int)(HIDWORD(v37) - v38) > 8 )
    goto LABEL_30;
  if ( *(_BYTE *)v12 > 0x1Cu )
  {
    v14 = *(_QWORD *)(v12 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17 <= 1 )
      v14 = **(_QWORD **)(v14 + 16);
    if ( sub_BCAC40(v14, 1) )
    {
      if ( *(_BYTE *)v12 == 57 )
      {
        if ( (*(_BYTE *)(v12 + 7) & 0x40) != 0 )
          v7 = *(__int64 **)(v12 - 8);
        else
          v7 = (__int64 *)(v12 - 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF));
        v15 = *v7;
        if ( *v7 )
        {
          v22 = v7[4];
          if ( v22 )
            goto LABEL_42;
        }
      }
      else
      {
        if ( *(_BYTE *)v12 != 86 )
          goto LABEL_18;
        v15 = *(_QWORD *)(v12 - 96);
        if ( *(_QWORD *)(v15 + 8) != *(_QWORD *)(v12 + 8) )
          goto LABEL_18;
        v26 = *(_BYTE **)(v12 - 32);
        if ( *v26 > 0x15u )
          goto LABEL_18;
        v22 = *(_QWORD *)(v12 - 64);
        if ( !sub_AC30F0((__int64)v26) || !v22 )
          goto LABEL_18;
LABEL_42:
        v23 = (unsigned int)v29;
        v24 = (unsigned int)v29 + 1LL;
        if ( v24 > HIDWORD(v29) )
        {
          sub_C8D5F0((__int64)&v28, v30, v24, 8u, (__int64)a5, a6);
          v23 = (unsigned int)v29;
        }
        v28[v23] = v22;
        v10 = HIDWORD(v29);
        LODWORD(v29) = v29 + 1;
        v25 = (unsigned int)v29;
        if ( (unsigned __int64)(unsigned int)v29 + 1 > HIDWORD(v29) )
        {
          sub_C8D5F0((__int64)&v28, v30, (unsigned int)v29 + 1LL, 8u, (__int64)a5, a6);
          v25 = (unsigned int)v29;
        }
        v7 = v28;
        v28[v25] = v15;
        LODWORD(v29) = v29 + 1;
      }
    }
  }
LABEL_18:
  v33 = v12;
  v31 = &v33;
  v32 = 0x400000001LL;
  if ( (unsigned __int8)(*(_BYTE *)v12 - 82) > 1u )
  {
    a5 = &v33;
    v16 = (__int64 *)&v34;
    goto LABEL_20;
  }
  sub_2A45340(v12, (__int64)&v31, (__int64)v7, v10, (__int64)a5, a6);
  a5 = v31;
  v16 = &v31[(unsigned int)v32];
  if ( v16 != v31 )
  {
LABEL_20:
    v17 = a5;
    do
    {
      while ( 1 )
      {
        v18 = *v17;
        if ( sub_2A45310(*v17) )
          break;
        if ( v16 == ++v17 )
          goto LABEL_26;
      }
      v19 = sub_22077B0(0x40u);
      if ( v19 )
      {
        *(_QWORD *)(v19 + 8) = 0;
        *(_QWORD *)(v19 + 16) = 0;
        *(_DWORD *)(v19 + 24) = 1;
        *(_QWORD *)(v19 + 32) = v18;
        *(_QWORD *)(v19 + 48) = v12;
        *(_QWORD *)v19 = &unk_4A22DE0;
        *(_QWORD *)(v19 + 56) = a2;
      }
      ++v17;
      sub_2A481C0(a1, a4, v18, v19, v20, v21);
    }
    while ( v16 != v17 );
LABEL_26:
    a5 = v31;
  }
  if ( a5 != &v33 )
    _libc_free((unsigned __int64)a5);
  v11 = v29;
  v10 = v39;
  if ( (_DWORD)v29 )
    goto LABEL_8;
LABEL_30:
  if ( !(_BYTE)v10 )
    _libc_free((unsigned __int64)v36);
  if ( v28 != v30 )
    _libc_free((unsigned __int64)v28);
}
