// Function: sub_2600E40
// Address: 0x2600e40
//
__int64 __fastcall sub_2600E40(__int64 *a1)
{
  __int64 v1; // rdx
  unsigned __int8 *v2; // r12
  unsigned int v3; // eax
  __int64 v4; // r13
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rdx
  __int64 result; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 v11; // rdx
  __int64 v12; // r13
  __int64 v13; // r15
  unsigned int v14; // r9d
  __int64 v15; // r10
  __int64 *v16; // rcx
  __int64 v17; // r11
  int v18; // ebx
  unsigned int v19; // eax
  __int64 *v20; // rsi
  __int64 v21; // rdi
  __int64 v22; // rdx
  char *v23; // rax
  __int64 v24; // rdx
  _BYTE *v25; // rsi
  _BYTE *v26; // rdx
  _QWORD *v27; // r14
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rdi
  int v31; // esi
  __int64 v32; // rax
  unsigned int v33; // ecx
  _QWORD *v34; // rax
  _QWORD *v35; // rdx
  _QWORD *v36; // rax
  unsigned __int64 v37; // rax
  _BYTE *v38; // rsi
  _QWORD *v39; // r12
  __int64 v40; // rdx
  unsigned int v41; // eax
  int v42; // r12d
  unsigned int v43; // eax
  int v44; // [rsp+Ch] [rbp-124h]
  __int64 *v45; // [rsp+10h] [rbp-120h]
  __int64 v46; // [rsp+20h] [rbp-110h]
  __int64 v48; // [rsp+38h] [rbp-F8h]
  unsigned __int64 v49; // [rsp+40h] [rbp-F0h]
  __int64 v50; // [rsp+48h] [rbp-E8h]
  __int64 v51; // [rsp+50h] [rbp-E0h] BYREF
  _QWORD *v52; // [rsp+58h] [rbp-D8h]
  __int64 v53; // [rsp+60h] [rbp-D0h]
  __int64 v54; // [rsp+68h] [rbp-C8h]
  _BYTE *v55; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v56; // [rsp+78h] [rbp-B8h]
  _BYTE v57[16]; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v58[2]; // [rsp+90h] [rbp-A0h] BYREF
  char v59; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v60[2]; // [rsp+B0h] [rbp-80h] BYREF
  char v61; // [rsp+C0h] [rbp-70h] BYREF
  _QWORD v62[4]; // [rsp+D0h] [rbp-60h] BYREF
  __int16 v63; // [rsp+F0h] [rbp-40h]

  v1 = *(_QWORD *)(*a1 + 16);
  v2 = *(unsigned __int8 **)(v1 + 16);
  v3 = *v2 - 30;
  if ( v3 <= 0xA )
  {
    v4 = *((_QWORD *)v2 + 5);
    v48 = 0;
    v5 = *(_QWORD *)(sub_B43CB0(*(_QWORD *)(v1 + 16)) + 72) & 0xFFFFFFFFFFFFFFF8LL;
    v6 = v5 - 24;
    if ( !v5 )
      v6 = 0;
    v3 = *v2 - 30;
    if ( v4 == v6 )
      goto LABEL_7;
    v1 = *(_QWORD *)(*a1 + 16);
  }
  v48 = *(_QWORD *)(*(_QWORD *)(v1 + 8) + 16LL);
LABEL_7:
  if ( v3 > 0xA )
  {
    result = sub_B46B10((__int64)v2, 0);
    if ( v48 != result )
      return result;
  }
  v51 = 0;
  v8 = *a1;
  v9 = *(_QWORD *)(*a1 + 8);
  v52 = 0;
  v10 = *(_QWORD *)(v9 + 16);
  v53 = 0;
  v54 = 0;
  v11 = *(_QWORD *)(v10 + 40);
  v12 = v10 + 24;
  a1[34] = v11;
  a1[33] = v11;
  sub_25FFEC0(*(_QWORD *)(v8 + 8), *(_QWORD *)(v8 + 16), (__int64)&v51);
  v13 = *((_QWORD *)v2 + 5);
  v45 = (__int64 *)(v10 + 24);
  a1[35] = v13;
  v49 = sub_986580(v13);
  v50 = 0;
  while ( 1 )
  {
    if ( *(_BYTE *)(v12 - 24) != 84 )
    {
      if ( *(_BYTE *)v10 == 84 )
      {
        v32 = *(_QWORD *)(a1[34] + 56);
        if ( !v32 || v10 != v32 - 24 )
        {
          v15 = (__int64)v52;
          v46 = (unsigned int)v54;
          return sub_C7D6A0(v15, v46 * 8, 8);
        }
      }
      if ( *v2 == 84 )
      {
        v36 = (_QWORD *)sub_AA5190(v13);
        if ( !v36 )
          BUG();
        v37 = *v36 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v37 || v2 != (unsigned __int8 *)(v37 - 24) )
          return sub_C7D6A0((__int64)v52, 8LL * (unsigned int)v54, 8);
      }
      v23 = (char *)sub_BD5D20(a1[33]);
      if ( v23 )
      {
        v55 = v57;
        sub_25F5F00((__int64 *)&v55, v23, (__int64)&v23[v24]);
        v25 = v55;
        v26 = &v55[v56];
      }
      else
      {
        v26 = v57;
        v57[0] = 0;
        v55 = v57;
        v25 = v57;
        v56 = 0;
      }
      v27 = (_QWORD *)a1[33];
      v58[0] = (__int64)&v59;
      sub_25F61C0(v58, v25, (__int64)v26);
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v58[1]) > 0xA )
      {
        sub_2241490((unsigned __int64 *)v58, "_to_outline", 0xBu);
        v63 = 260;
        v62[0] = v58;
        a1[34] = sub_AA8550(v27, v45, 0, (__int64)v62, 0);
        sub_2240A30((unsigned __int64 *)v58);
        sub_AA5DE0(a1[33], a1[33], a1[34]);
        if ( v50 )
          sub_AA5DE0(a1[33], v50, a1[33]);
        *((_BYTE *)a1 + 256) = 1;
        if ( (unsigned int)*v2 - 30 <= 0xA )
        {
          v28 = *((_QWORD *)v2 + 5);
          *((_BYTE *)a1 + 129) = 1;
          a1[36] = 0;
          a1[35] = v28;
          goto LABEL_32;
        }
        v38 = v55;
        v60[0] = (__int64)&v61;
        v39 = *(_QWORD **)(v48 + 40);
        v40 = (__int64)&v55[v56];
        a1[35] = (__int64)v39;
        sub_25F61C0(v60, v38, v40);
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v60[1]) > 0xD )
        {
          sub_2241490((unsigned __int64 *)v60, "_after_outline", 0xEu);
          v63 = 260;
          v62[0] = v60;
          a1[36] = sub_AA8550(v39, (__int64 *)(v48 + 24), 0, (__int64)v62, 0);
          sub_2240A30((unsigned __int64 *)v60);
          sub_AA5DE0(a1[35], a1[35], a1[36]);
          sub_AA5DE0(a1[36], a1[33], a1[36]);
LABEL_32:
          ++v51;
          if ( !(_DWORD)v53 )
          {
            if ( !HIDWORD(v53) )
              goto LABEL_38;
            v29 = (unsigned int)v54;
            if ( (unsigned int)v54 > 0x40 )
            {
              sub_C7D6A0((__int64)v52, 8LL * (unsigned int)v54, 8);
              LODWORD(v54) = 0;
LABEL_36:
              v52 = 0;
LABEL_37:
              v53 = 0;
              goto LABEL_38;
            }
LABEL_53:
            v34 = v52;
            v35 = &v52[v29];
            if ( v52 == v35 )
              goto LABEL_37;
            do
              *v34++ = -4096;
            while ( v35 != v34 );
            v53 = 0;
LABEL_38:
            sub_25FFEC0(*(_QWORD *)(*a1 + 8), *(_QWORD *)(*a1 + 16), (__int64)&v51);
            sub_25F7360(a1[34], a1[33], a1[34], (__int64)&v51);
            v30 = a1[36];
            if ( v30 )
              sub_25F7360(v30, a1[35], a1[36], (__int64)&v51);
            sub_2240A30((unsigned __int64 *)&v55);
            return sub_C7D6A0((__int64)v52, 8LL * (unsigned int)v54, 8);
          }
          v33 = 4 * v53;
          v29 = (unsigned int)v54;
          if ( (unsigned int)(4 * v53) < 0x40 )
            v33 = 64;
          if ( v33 >= (unsigned int)v54 )
            goto LABEL_53;
          if ( (_DWORD)v53 == 1 )
          {
            v42 = 64;
          }
          else
          {
            _BitScanReverse(&v41, v53 - 1);
            v42 = 1 << (33 - (v41 ^ 0x1F));
            if ( v42 < 64 )
              v42 = 64;
            if ( v42 == (_DWORD)v54 )
              goto LABEL_69;
          }
          sub_C7D6A0((__int64)v52, 8LL * (unsigned int)v54, 8);
          v43 = sub_25F87F0(v42);
          LODWORD(v54) = v43;
          if ( !v43 )
            goto LABEL_36;
          v52 = (_QWORD *)sub_C7D670(8LL * v43, 8);
LABEL_69:
          sub_25FE760((__int64)&v51);
          goto LABEL_38;
        }
      }
      sub_4262D8((__int64)"basic_string::append");
    }
    if ( (*(_DWORD *)(v12 - 20) & 0x7FFFFFF) != 0 )
      break;
LABEL_21:
    v12 = *(_QWORD *)(v12 + 8);
    if ( !v12 )
      BUG();
  }
  v14 = 0;
  v15 = (__int64)v52;
  v16 = (__int64 *)(*(_QWORD *)(v12 - 32) + 32LL * *(unsigned int *)(v12 + 48));
  v46 = (unsigned int)v54;
  v17 = (__int64)&v16[*(_DWORD *)(v12 - 20) & 0x7FFFFFF];
  v18 = v54 - 1;
  do
  {
    while ( 1 )
    {
      v22 = *v16;
      if ( !(_DWORD)v54 )
        goto LABEL_19;
      v19 = v18 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v20 = &v52[v19];
      v21 = *v20;
      if ( v22 != *v20 )
        break;
LABEL_14:
      if ( v20 == &v52[v46] || v13 == v22 && (unsigned __int8 *)v49 != v2 )
        goto LABEL_19;
      if ( (__int64 *)v17 == ++v16 )
        goto LABEL_20;
    }
    v31 = 1;
    while ( v21 != -4096 )
    {
      v19 = v18 & (v31 + v19);
      v44 = v31 + 1;
      v20 = &v52[v19];
      v21 = *v20;
      if ( v22 == *v20 )
        goto LABEL_14;
      v31 = v44;
    }
LABEL_19:
    ++v16;
    v50 = v22;
    ++v14;
  }
  while ( (__int64 *)v17 != v16 );
LABEL_20:
  if ( v14 <= 1 )
    goto LABEL_21;
  return sub_C7D6A0(v15, v46 * 8, 8);
}
