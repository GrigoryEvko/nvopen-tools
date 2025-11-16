// Function: sub_D29180
// Address: 0xd29180
//
__int64 __fastcall sub_D29180(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  bool v8; // zf
  __int64 v9; // rax
  _QWORD *v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // r13
  __int64 v17; // rsi
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 *v20; // r15
  __int64 *v21; // r12
  _BYTE *v22; // r14
  _QWORD *v23; // rax
  char **v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 *v27; // r15
  __int64 *v28; // r13
  __int64 v29; // r14
  unsigned __int64 v30; // rax
  __int64 v32; // rax
  unsigned __int64 v33; // rdx
  unsigned __int64 v34; // rax
  __int64 v35; // [rsp+8h] [rbp-1E8h]
  __int64 v36; // [rsp+18h] [rbp-1D8h]
  __int64 v38; // [rsp+30h] [rbp-1C0h]
  __int64 v39; // [rsp+48h] [rbp-1A8h] BYREF
  __int64 v40; // [rsp+50h] [rbp-1A0h] BYREF
  char *v41; // [rsp+58h] [rbp-198h]
  __int64 v42; // [rsp+60h] [rbp-190h]
  int v43; // [rsp+68h] [rbp-188h]
  char v44; // [rsp+6Ch] [rbp-184h]
  char v45; // [rsp+70h] [rbp-180h] BYREF
  _BYTE *v46; // [rsp+90h] [rbp-160h] BYREF
  __int64 v47; // [rsp+98h] [rbp-158h]
  _BYTE v48[128]; // [rsp+A0h] [rbp-150h] BYREF
  char *v49; // [rsp+120h] [rbp-D0h] BYREF
  __int64 v50; // [rsp+128h] [rbp-C8h]
  __int64 v51; // [rsp+130h] [rbp-C0h] BYREF
  int v52; // [rsp+138h] [rbp-B8h]
  char v53; // [rsp+13Ch] [rbp-B4h]
  char v54; // [rsp+140h] [rbp-B0h] BYREF
  __int64 v55; // [rsp+150h] [rbp-A0h]
  __int64 v56; // [rsp+158h] [rbp-98h]
  __int64 v57; // [rsp+160h] [rbp-90h]
  int v58; // [rsp+168h] [rbp-88h]
  int v59; // [rsp+16Ch] [rbp-84h]

  v7 = a1 + 24;
  v8 = *(_BYTE *)(a1 + 104) == 0;
  v59 = 0;
  v49 = (char *)&v51;
  v50 = 0x400000000LL;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v36 = v7;
  if ( v8 )
  {
    v55 = 1;
    *(_QWORD *)(a1 + 32) = 0x400000000LL;
    *(_QWORD *)(a1 + 24) = a1 + 40;
    *(_QWORD *)(a1 + 72) = 1;
    *(_QWORD *)(a1 + 80) = 0;
    *(_QWORD *)(a1 + 88) = 0;
    *(_DWORD *)(a1 + 96) = 0;
    *(_BYTE *)(a1 + 104) = 1;
  }
  else
  {
    sub_D23360(v7, &v49, a3, a4, a5, a6);
    sub_C7D6A0(*(_QWORD *)(a1 + 80), 16LL * *(unsigned int *)(a1 + 96), 8);
    v9 = v56;
    ++*(_QWORD *)(a1 + 72);
    ++v55;
    *(_QWORD *)(a1 + 80) = v9;
    v56 = 0;
    *(_QWORD *)(a1 + 88) = v57;
    v57 = 0;
    *(_DWORD *)(a1 + 96) = v58;
    v58 = 0;
  }
  sub_C7D6A0(0, 0, 8);
  if ( v49 != (char *)&v51 )
    _libc_free(v49, 0);
  v49 = 0;
  v46 = v48;
  v47 = 0x1000000000LL;
  v41 = &v45;
  v50 = (__int64)&v54;
  v51 = 16;
  v14 = *(_QWORD *)(a1 + 8);
  v52 = 0;
  v53 = 1;
  v15 = *(_QWORD *)(v14 + 80);
  v40 = 0;
  v42 = 4;
  v43 = 0;
  v44 = 1;
  v38 = v15;
  v35 = v14 + 72;
  if ( v15 != v14 + 72 )
  {
    while ( 1 )
    {
      if ( !v38 )
        BUG();
      v16 = *(_QWORD *)(v38 + 32);
      if ( v16 != v38 + 24 )
        break;
LABEL_25:
      v38 = *(_QWORD *)(v38 + 8);
      if ( v35 == v38 )
        goto LABEL_26;
    }
    while ( 1 )
    {
      if ( !v16 )
        BUG();
      if ( (unsigned __int8)(*(_BYTE *)(v16 - 24) - 34) <= 0x33u )
      {
        v17 = 0x8000000000041LL;
        if ( _bittest64(&v17, (unsigned int)*(unsigned __int8 *)(v16 - 24) - 34) )
        {
          v18 = *(_QWORD *)(v16 - 56);
          if ( v18 )
          {
            if ( !*(_BYTE *)v18 && *(_QWORD *)(v18 + 24) == *(_QWORD *)(v16 + 56) && !sub_B2FC80(*(_QWORD *)(v16 - 56)) )
            {
              sub_AE6EC0((__int64)&v40, v18);
              if ( (_BYTE)v10 )
              {
                sub_AE6EC0((__int64)&v49, v18);
                v34 = sub_D29010(*(_QWORD *)a1, v18);
                sub_D25660(v36, a1 + 72, v34, 1u);
              }
            }
          }
        }
      }
      v19 = 32LL * (*(_DWORD *)(v16 - 20) & 0x7FFFFFF);
      if ( (*(_BYTE *)(v16 - 17) & 0x40) != 0 )
      {
        v12 = *(_QWORD *)(v16 - 32);
        v20 = (__int64 *)(v12 + v19);
      }
      else
      {
        v20 = (__int64 *)(v16 - 24);
        v12 = v16 - 24 - v19;
      }
      v21 = (__int64 *)v12;
      if ( (__int64 *)v12 != v20 )
        break;
LABEL_24:
      v16 = *(_QWORD *)(v16 + 8);
      if ( v38 + 24 == v16 )
        goto LABEL_25;
    }
    while ( 1 )
    {
      v22 = (_BYTE *)*v21;
      if ( *(_BYTE *)*v21 > 0x15u )
        goto LABEL_23;
      if ( v53 )
      {
        v23 = (_QWORD *)v50;
        v11 = HIDWORD(v51);
        v10 = (_QWORD *)(v50 + 8LL * HIDWORD(v51));
        if ( (_QWORD *)v50 != v10 )
        {
          while ( v22 != (_BYTE *)*v23 )
          {
            if ( v10 == ++v23 )
              goto LABEL_41;
          }
          goto LABEL_23;
        }
LABEL_41:
        if ( HIDWORD(v51) < (unsigned int)v51 )
        {
          ++HIDWORD(v51);
          *v10 = v22;
          ++v49;
          goto LABEL_37;
        }
      }
      sub_C8CC70((__int64)&v49, *v21, (__int64)v10, v11, v12, v13);
      if ( (_BYTE)v10 )
      {
LABEL_37:
        v32 = (unsigned int)v47;
        v11 = HIDWORD(v47);
        v33 = (unsigned int)v47 + 1LL;
        if ( v33 > HIDWORD(v47) )
        {
          sub_C8D5F0((__int64)&v46, v48, v33, 8u, v12, v13);
          v32 = (unsigned int)v47;
        }
        v10 = v46;
        v21 += 4;
        *(_QWORD *)&v46[8 * v32] = v22;
        LODWORD(v47) = v47 + 1;
        if ( v20 == v21 )
          goto LABEL_24;
      }
      else
      {
LABEL_23:
        v21 += 4;
        if ( v20 == v21 )
          goto LABEL_24;
      }
    }
  }
LABEL_26:
  v24 = &v49;
  v39 = a1;
  sub_D24710((__int64)&v46, (__int64)&v49, (void (__fastcall *)(__int64, __int64))sub_D296F0, (__int64)&v39, v12);
  v27 = *(__int64 **)(*(_QWORD *)a1 + 640LL);
  v28 = &v27[*(unsigned int *)(*(_QWORD *)a1 + 648LL)];
  while ( v28 != v27 )
  {
    while ( 1 )
    {
      v29 = *v27;
      v24 = (char **)*v27;
      if ( !(unsigned __int8)sub_B19060((__int64)&v49, *v27, v25, v26) )
        break;
      if ( v28 == ++v27 )
        goto LABEL_31;
    }
    ++v27;
    v30 = sub_D29010(*(_QWORD *)a1, v29);
    v24 = (char **)(a1 + 72);
    sub_D25660(v36, a1 + 72, v30, 0);
  }
LABEL_31:
  if ( v53 )
  {
    if ( v44 )
      goto LABEL_33;
  }
  else
  {
    _libc_free(v50, v24);
    if ( v44 )
      goto LABEL_33;
  }
  _libc_free(v41, v24);
LABEL_33:
  if ( v46 != v48 )
    _libc_free(v46, v24);
  return v36;
}
