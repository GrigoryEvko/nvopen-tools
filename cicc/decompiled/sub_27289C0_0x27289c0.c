// Function: sub_27289C0
// Address: 0x27289c0
//
void __fastcall sub_27289C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 v8; // r15
  __int64 v9; // rcx
  __int64 v10; // rdx
  char *v11; // rax
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  int v14; // eax
  _BYTE *v15; // rdi
  _BYTE *v16; // rsi
  __int64 v17; // r15
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdx
  __int64 v22; // r8
  __int64 v23; // r9
  unsigned int v24; // r14d
  unsigned __int64 v25; // r13
  __int64 v26; // rcx
  bool v27; // r14
  __int64 v28; // r14
  __int64 v29; // r15
  char *v30; // rax
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  unsigned __int64 v33; // [rsp+10h] [rbp-170h] BYREF
  unsigned int v34; // [rsp+18h] [rbp-168h]
  _BYTE *v35; // [rsp+20h] [rbp-160h] BYREF
  __int64 v36; // [rsp+28h] [rbp-158h]
  _BYTE v37[128]; // [rsp+30h] [rbp-150h] BYREF
  __int64 v38; // [rsp+B0h] [rbp-D0h] BYREF
  char *v39; // [rsp+B8h] [rbp-C8h]
  __int64 v40; // [rsp+C0h] [rbp-C0h]
  int v41; // [rsp+C8h] [rbp-B8h]
  char v42; // [rsp+CCh] [rbp-B4h]
  char v43; // [rsp+D0h] [rbp-B0h] BYREF

  v39 = &v43;
  v7 = *(_QWORD *)(a1 + 16);
  v35 = v37;
  v38 = 0;
  v40 = 16;
  v41 = 0;
  v42 = 1;
  v36 = 0x1000000000LL;
  if ( v7 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v8 = *(_QWORD *)(v7 + 24);
        v9 = *(_QWORD *)(v8 + 8);
        v10 = *(unsigned __int8 *)(v9 + 8);
        if ( (unsigned int)(v10 - 17) <= 1 )
          v10 = *(unsigned __int8 *)(**(_QWORD **)(v9 + 16) + 8LL);
        if ( (_BYTE)v10 == 12 )
          break;
        v7 = *(_QWORD *)(v7 + 8);
        if ( !v7 )
          goto LABEL_15;
      }
      if ( !v42 )
        goto LABEL_46;
      v11 = v39;
      v9 = HIDWORD(v40);
      v10 = (__int64)&v39[8 * HIDWORD(v40)];
      if ( v39 != (char *)v10 )
      {
        while ( v8 != *(_QWORD *)v11 )
        {
          v11 += 8;
          if ( (char *)v10 == v11 )
            goto LABEL_45;
        }
        goto LABEL_12;
      }
LABEL_45:
      if ( HIDWORD(v40) < (unsigned int)v40 )
      {
        ++HIDWORD(v40);
        *(_QWORD *)v10 = v8;
        ++v38;
      }
      else
      {
LABEL_46:
        sub_C8CC70((__int64)&v38, *(_QWORD *)(v7 + 24), v10, v9, a5, a6);
      }
LABEL_12:
      v12 = (unsigned int)v36;
      v13 = (unsigned int)v36 + 1LL;
      if ( v13 > HIDWORD(v36) )
      {
        sub_C8D5F0((__int64)&v35, v37, v13, 8u, a5, a6);
        v12 = (unsigned int)v36;
      }
      *(_QWORD *)&v35[8 * v12] = v8;
      LODWORD(v36) = v36 + 1;
      v7 = *(_QWORD *)(v7 + 8);
      if ( !v7 )
      {
LABEL_15:
        v14 = v36;
        v15 = v35;
        goto LABEL_16;
      }
    }
  }
  v15 = v37;
  v14 = 0;
LABEL_16:
  v16 = &v35;
  if ( v14 )
  {
    while ( 1 )
    {
      v17 = *(_QWORD *)&v15[8 * v14 - 8];
      LODWORD(v36) = v14 - 1;
      sub_B44F30((unsigned __int8 *)v17);
      sub_B44B50((__int64 *)v17, (__int64)v16);
      sub_B44A60(v17);
      v16 = (_BYTE *)a2;
      sub_D19730((__int64)&v33, a2, v17, v18, v19, v20);
      v24 = v34;
      if ( v34 )
      {
        v25 = v33;
        if ( v34 <= 0x40 )
        {
          v26 = 64 - v34;
          v27 = v33 == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v34);
        }
        else
        {
          v27 = v24 == (unsigned int)sub_C445E0((__int64)&v33);
          if ( v25 )
            j_j___libc_free_0_0(v25);
        }
        if ( !v27 )
        {
          v28 = *(_QWORD *)(v17 + 16);
          if ( v28 )
            break;
        }
      }
LABEL_29:
      v14 = v36;
      v15 = v35;
      if ( !(_DWORD)v36 )
        goto LABEL_30;
    }
    while ( 1 )
    {
      v29 = *(_QWORD *)(v28 + 24);
      if ( !v42 )
        goto LABEL_34;
      v30 = v39;
      v26 = HIDWORD(v40);
      v21 = (__int64)&v39[8 * HIDWORD(v40)];
      if ( v39 != (char *)v21 )
      {
        while ( v29 != *(_QWORD *)v30 )
        {
          v30 += 8;
          if ( (char *)v21 == v30 )
            goto LABEL_42;
        }
        goto LABEL_28;
      }
LABEL_42:
      if ( HIDWORD(v40) < (unsigned int)v40 )
      {
        ++HIDWORD(v40);
        *(_QWORD *)v21 = v29;
        ++v38;
LABEL_35:
        v26 = *(_QWORD *)(v29 + 8);
        v21 = *(unsigned __int8 *)(v26 + 8);
        if ( (unsigned int)(v21 - 17) <= 1 )
          v21 = *(unsigned __int8 *)(**(_QWORD **)(v26 + 16) + 8LL);
        if ( (_BYTE)v21 != 12 )
          goto LABEL_28;
        v31 = (unsigned int)v36;
        v26 = HIDWORD(v36);
        v32 = (unsigned int)v36 + 1LL;
        if ( v32 > HIDWORD(v36) )
        {
          v16 = v37;
          sub_C8D5F0((__int64)&v35, v37, v32, 8u, v22, v23);
          v31 = (unsigned int)v36;
        }
        v21 = (__int64)v35;
        *(_QWORD *)&v35[8 * v31] = v29;
        LODWORD(v36) = v36 + 1;
        v28 = *(_QWORD *)(v28 + 8);
        if ( !v28 )
          goto LABEL_29;
      }
      else
      {
LABEL_34:
        v16 = *(_BYTE **)(v28 + 24);
        sub_C8CC70((__int64)&v38, (__int64)v16, v21, v26, v22, v23);
        if ( (_BYTE)v21 )
          goto LABEL_35;
LABEL_28:
        v28 = *(_QWORD *)(v28 + 8);
        if ( !v28 )
          goto LABEL_29;
      }
    }
  }
LABEL_30:
  if ( v15 != v37 )
    _libc_free((unsigned __int64)v15);
  if ( !v42 )
    _libc_free((unsigned __int64)v39);
}
