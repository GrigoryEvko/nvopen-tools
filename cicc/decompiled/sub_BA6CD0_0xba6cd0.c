// Function: sub_BA6CD0
// Address: 0xba6cd0
//
__int64 __fastcall sub_BA6CD0(__int64 a1, __int64 *a2)
{
  __int64 *v2; // r14
  _BYTE *v4; // rax
  __int64 v5; // rdx
  _QWORD *v6; // rcx
  _QWORD *v7; // r15
  _BYTE *v8; // rdx
  unsigned __int8 v9; // al
  __int64 v10; // rax
  __int64 **v11; // rax
  __int64 **v12; // rdx
  _BYTE *v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rdx
  _QWORD *v16; // r14
  _QWORD *v17; // r15
  _BYTE *v18; // rdx
  unsigned __int8 v19; // al
  _BYTE *v20; // rax
  __int64 **v21; // rax
  __int64 **v22; // rdx
  __int64 **v23; // rax
  __int64 **v24; // rdx
  _BYTE *v25; // rax
  __int64 **v26; // rcx
  __int64 v27; // rdx
  _QWORD *v28; // r15
  _QWORD *v29; // r13
  _BYTE *v30; // rdx
  unsigned __int8 v31; // al
  __int64 **v32; // rax
  __int64 **v33; // rax
  __int64 v34; // r13
  __int64 *v35; // rdi
  __int64 v36; // rsi
  __int64 *v38; // rdi
  __int64 v39; // rax
  _QWORD *v40; // [rsp+0h] [rbp-1E0h]
  _BYTE *v41; // [rsp+18h] [rbp-1C8h] BYREF
  __int64 v42; // [rsp+20h] [rbp-1C0h] BYREF
  __int64 v43; // [rsp+28h] [rbp-1B8h]
  __int64 v44; // [rsp+30h] [rbp-1B0h]
  __int64 v45; // [rsp+38h] [rbp-1A8h]
  __int64 *v46; // [rsp+40h] [rbp-1A0h]
  __int64 v47; // [rsp+48h] [rbp-198h]
  _BYTE v48[32]; // [rsp+50h] [rbp-190h] BYREF
  __int64 v49; // [rsp+70h] [rbp-170h] BYREF
  __int64 **v50; // [rsp+78h] [rbp-168h]
  __int64 v51; // [rsp+80h] [rbp-160h]
  int v52; // [rsp+88h] [rbp-158h]
  char v53; // [rsp+8Ch] [rbp-154h]
  char v54; // [rsp+90h] [rbp-150h] BYREF
  __int64 v55; // [rsp+110h] [rbp-D0h] BYREF
  __int64 **v56; // [rsp+118h] [rbp-C8h]
  __int64 v57; // [rsp+120h] [rbp-C0h]
  int v58; // [rsp+128h] [rbp-B8h]
  char v59; // [rsp+12Ch] [rbp-B4h]
  char v60; // [rsp+130h] [rbp-B0h] BYREF

  if ( !a1 )
    return 0;
  v2 = a2;
  if ( !a2 )
    return 0;
  v49 = 0;
  v50 = (__int64 **)&v54;
  v56 = (__int64 **)&v60;
  v51 = 16;
  v52 = 0;
  v53 = 1;
  v55 = 0;
  v57 = 16;
  v58 = 0;
  v59 = 1;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = (__int64 *)v48;
  v47 = 0x400000000LL;
  v4 = sub_A17150((_BYTE *)(a1 - 16));
  v6 = &v4[8 * v5];
  v7 = v4;
  if ( v6 != (_QWORD *)v4 )
  {
    while ( 1 )
    {
      v8 = (_BYTE *)*v7;
      if ( (unsigned __int8)(*(_BYTE *)*v7 - 5) > 0x1Fu )
        goto LABEL_15;
      v9 = *(v8 - 16);
      if ( (v9 & 2) != 0 )
      {
        if ( *((_DWORD *)v8 - 6) <= 1u )
          goto LABEL_15;
        v10 = *((_QWORD *)v8 - 4);
LABEL_8:
        a2 = *(__int64 **)(v10 + 8);
        if ( !a2 || (unsigned __int8)(*(_BYTE *)a2 - 5) > 0x1Fu )
          goto LABEL_15;
        if ( v53 )
        {
          v11 = v50;
          v12 = &v50[HIDWORD(v51)];
          if ( v50 != v12 )
          {
            while ( a2 != *v11 )
            {
              if ( v12 == ++v11 )
                goto LABEL_63;
            }
            goto LABEL_15;
          }
LABEL_63:
          if ( HIDWORD(v51) >= (unsigned int)v51 )
            goto LABEL_64;
          ++v7;
          ++HIDWORD(v51);
          *v12 = a2;
          ++v49;
          if ( v6 == v7 )
            break;
        }
        else
        {
LABEL_64:
          v40 = v6;
          ++v7;
          sub_C8CC70(&v49, a2);
          v6 = v40;
          if ( v40 == v7 )
            break;
        }
      }
      else
      {
        a2 = (__int64 *)((*((_WORD *)v8 - 8) >> 6) & 0xF);
        if ( ((*((_WORD *)v8 - 8) >> 6) & 0xFu) > 1 )
        {
          v10 = (__int64)&v8[-8 * ((v9 >> 2) & 0xF) - 16];
          goto LABEL_8;
        }
LABEL_15:
        if ( v6 == ++v7 )
          break;
      }
    }
  }
  v13 = sub_A17150((_BYTE *)v2 - 16);
  v16 = &v13[8 * v15];
  v17 = v13;
  if ( v16 != (_QWORD *)v13 )
  {
    while ( 1 )
    {
      v18 = (_BYTE *)*v17;
      if ( (unsigned __int8)(*(_BYTE *)*v17 - 5) > 0x1Fu )
        goto LABEL_36;
      v19 = *(v18 - 16);
      if ( (v19 & 2) != 0 )
      {
        if ( *((_DWORD *)v18 - 6) > 1u )
        {
          v20 = (_BYTE *)*((_QWORD *)v18 - 4);
          goto LABEL_21;
        }
LABEL_36:
        if ( v16 == ++v17 )
          break;
      }
      else
      {
        a2 = (__int64 *)((*((_WORD *)v18 - 8) >> 6) & 0xF);
        if ( ((*((_WORD *)v18 - 8) >> 6) & 0xFu) <= 1 )
          goto LABEL_36;
        v18 -= 8 * ((v19 >> 2) & 0xF);
        v20 = v18 - 16;
LABEL_21:
        a2 = (__int64 *)*((_QWORD *)v20 + 1);
        if ( !a2 || (unsigned __int8)(*(_BYTE *)a2 - 5) > 0x1Fu )
          goto LABEL_36;
        if ( v53 )
        {
          v21 = v50;
          v22 = &v50[HIDWORD(v51)];
          if ( v50 == v22 )
            goto LABEL_36;
          while ( a2 != *v21 )
          {
            if ( v22 == ++v21 )
              goto LABEL_36;
          }
LABEL_28:
          if ( !v59 )
            goto LABEL_34;
          v23 = v56;
          v24 = &v56[HIDWORD(v57)];
          if ( v56 != v24 )
          {
            while ( a2 != *v23 )
            {
              if ( v24 == ++v23 )
                goto LABEL_81;
            }
            goto LABEL_35;
          }
LABEL_81:
          if ( HIDWORD(v57) < (unsigned int)v57 )
          {
            ++HIDWORD(v57);
            *v24 = a2;
            ++v55;
          }
          else
          {
LABEL_34:
            sub_C8CC70(&v55, a2);
          }
LABEL_35:
          a2 = (__int64 *)&v41;
          v41 = (_BYTE *)*v17;
          sub_BA67F0((__int64)&v42, (__int64 *)&v41);
          goto LABEL_36;
        }
        if ( sub_C8CA60(&v49, a2, v18, v14) )
          goto LABEL_28;
        if ( v16 == ++v17 )
          break;
      }
    }
  }
  v25 = sub_A17150((_BYTE *)(a1 - 16));
  v28 = &v25[8 * v27];
  v29 = v25;
  if ( v25 != (_BYTE *)v28 )
  {
    do
    {
      v30 = (_BYTE *)*v29;
      if ( (unsigned __int8)(*(_BYTE *)*v29 - 5) <= 0x1Fu )
      {
        v31 = *(v30 - 16);
        if ( (v31 & 2) != 0 )
        {
          if ( *((_DWORD *)v30 - 6) <= 1u )
            goto LABEL_50;
          v32 = (__int64 **)*((_QWORD *)v30 - 4);
        }
        else
        {
          v26 = (__int64 **)((*((_WORD *)v30 - 8) >> 6) & 0xF);
          if ( ((*((_WORD *)v30 - 8) >> 6) & 0xFu) <= 1 )
            goto LABEL_50;
          v26 = (__int64 **)&v30[-8 * ((v31 >> 2) & 0xF)];
          v32 = v26 - 2;
        }
        a2 = v32[1];
        if ( !a2 || (unsigned __int8)(*(_BYTE *)a2 - 5) > 0x1Fu )
          goto LABEL_50;
        if ( v59 )
        {
          v33 = v56;
          v26 = &v56[HIDWORD(v57)];
          if ( v56 == v26 )
            goto LABEL_50;
          while ( a2 != *v33 )
          {
            if ( v26 == ++v33 )
              goto LABEL_50;
          }
        }
        else
        {
          if ( !sub_C8CA60(&v55, a2, v30, v26) )
            goto LABEL_50;
          v30 = (_BYTE *)*v29;
        }
        a2 = (__int64 *)&v41;
        v41 = v30;
        sub_BA67F0((__int64)&v42, (__int64 *)&v41);
      }
LABEL_50:
      ++v29;
    }
    while ( v28 != v29 );
  }
  v34 = 0;
  if ( !(_DWORD)v47 )
  {
    v35 = v46;
    if ( v46 == (__int64 *)v48 )
      goto LABEL_54;
    goto LABEL_53;
  }
  a2 = v46;
  v38 = (__int64 *)(*(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a1 + 8) & 4) != 0 )
    v38 = (__int64 *)*v38;
  v39 = sub_B9D9A0(v38, v46, (__int64 *)(unsigned int)v47);
  v35 = v46;
  v34 = v39;
  if ( v46 != (__int64 *)v48 )
LABEL_53:
    _libc_free(v35, a2);
LABEL_54:
  v36 = 8LL * (unsigned int)v45;
  sub_C7D6A0(v43, v36, 8);
  if ( !v59 )
  {
    _libc_free(v56, v36);
    if ( v53 )
      return v34;
    goto LABEL_77;
  }
  if ( !v53 )
LABEL_77:
    _libc_free(v50, v36);
  return v34;
}
