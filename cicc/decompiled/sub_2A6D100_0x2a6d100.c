// Function: sub_2A6D100
// Address: 0x2a6d100
//
void __fastcall sub_2A6D100(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // ebx
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 *v10; // rax
  __int64 *v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r12
  __int64 v17; // rdx
  const void **v18; // r14
  __int64 v19; // rax
  const void **v20; // rax
  unsigned int v21; // r14d
  unsigned __int64 v22; // r15
  __int64 v23; // rbx
  unsigned __int8 *v24; // rax
  unsigned __int8 *v25; // rdx
  unsigned __int8 v26; // al
  unsigned __int64 v27; // rdx
  unsigned int v28; // eax
  unsigned __int64 v29; // r15
  unsigned __int8 *v30; // rax
  __int64 v31; // rcx
  int v32; // edx
  unsigned __int8 *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int16 v36; // [rsp+12h] [rbp-EEh]
  unsigned __int8 *v37; // [rsp+18h] [rbp-E8h]
  unsigned __int8 *v38; // [rsp+18h] [rbp-E8h]
  __int64 v39; // [rsp+28h] [rbp-D8h]
  __int64 v40; // [rsp+38h] [rbp-C8h]
  int v41; // [rsp+40h] [rbp-C0h]
  unsigned __int8 *v42; // [rsp+40h] [rbp-C0h]
  unsigned __int8 **v43; // [rsp+48h] [rbp-B8h]
  __int64 v44; // [rsp+50h] [rbp-B0h] BYREF
  int v45; // [rsp+58h] [rbp-A8h]
  __int64 v46; // [rsp+60h] [rbp-A0h] BYREF
  int v47; // [rsp+68h] [rbp-98h]
  __int16 v48; // [rsp+70h] [rbp-90h] BYREF
  unsigned __int64 v49; // [rsp+78h] [rbp-88h] BYREF
  unsigned int v50; // [rsp+80h] [rbp-80h]
  unsigned __int64 v51; // [rsp+88h] [rbp-78h] BYREF
  unsigned int v52; // [rsp+90h] [rbp-70h]
  __int64 v53; // [rsp+A0h] [rbp-60h] BYREF
  unsigned __int64 v54; // [rsp+A8h] [rbp-58h] BYREF
  __int64 v55; // [rsp+B0h] [rbp-50h] BYREF
  unsigned __int64 v56; // [rsp+B8h] [rbp-48h] BYREF
  unsigned int v57; // [rsp+C0h] [rbp-40h]

  v8 = *(_QWORD *)(a2 - 32);
  v40 = v8;
  if ( v8 )
  {
    if ( *(_BYTE *)v8 )
    {
      v40 = 0;
    }
    else
    {
      v9 = 0;
      if ( *(_QWORD *)(a2 + 80) == *(_QWORD *)(v8 + 24) )
        v9 = *(_QWORD *)(a2 - 32);
      v40 = v9;
    }
  }
  if ( *(_BYTE *)(a1 + 708) )
  {
    v10 = *(__int64 **)(a1 + 688);
    v11 = &v10[*(unsigned int *)(a1 + 700)];
    if ( v10 == v11 )
      return;
    v12 = v40;
    while ( v40 != *v10 )
    {
      if ( v11 == ++v10 )
        return;
    }
  }
  else if ( !sub_C8CA60(a1 + 680, v40) )
  {
    return;
  }
  v13 = *(_QWORD *)(v40 + 80);
  if ( v13 )
    v13 -= 24;
  sub_2A62EB0(a1, v13, v11, v12, a5, a6);
  v43 = (unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  if ( (*(_BYTE *)(v40 + 2) & 1) != 0 )
  {
    sub_B2C6D0(v40, v13, v14, v15);
    v16 = *(_QWORD *)(v40 + 96);
    if ( (*(_BYTE *)(v40 + 2) & 1) != 0 )
      sub_B2C6D0(v40, v13, v34, v35);
    v17 = *(_QWORD *)(v40 + 96);
  }
  else
  {
    v16 = *(_QWORD *)(v40 + 96);
    v17 = v16;
  }
  v39 = v17 + 40LL * *(_QWORD *)(v40 + 104);
  if ( v16 != v39 )
  {
    v18 = (const void **)&v53;
    while ( (unsigned __int8)sub_B2D680(v16) && !(unsigned __int8)sub_B2DCE0(v40) )
    {
      sub_2A6A450(a1, v16);
LABEL_40:
      v43 += 4;
      v16 += 40;
      if ( v16 == v39 )
        return;
    }
    v19 = *(_QWORD *)(v16 + 8);
    if ( *(_BYTE *)(v19 + 8) != 15 )
    {
      HIWORD(v28) = v36;
      LOWORD(v28) = 256;
      v29 = v28 | ((unsigned __int64)(unsigned int)qword_500BEC8 << 32);
      v30 = (unsigned __int8 *)sub_2A68BC0(a1, *v43);
      v31 = *(_QWORD *)(v16 + 8);
      v42 = v30;
      v32 = *(unsigned __int8 *)(v31 + 8);
      if ( (unsigned int)(v32 - 17) <= 1 )
        LOBYTE(v32) = *(_BYTE *)(**(_QWORD **)(v31 + 16) + 8LL);
      if ( (_BYTE)v32 == 12 && (sub_B2D8F0((__int64)v18, v16), (_BYTE)v57) )
      {
        v45 = v54;
        if ( (unsigned int)v54 > 0x40 )
          sub_C43780((__int64)&v44, v18);
        else
          v44 = v53;
        v47 = v56;
        if ( (unsigned int)v56 > 0x40 )
          sub_C43780((__int64)&v46, (const void **)&v55);
        else
          v46 = v55;
        sub_22C06B0((__int64)&v48, (__int64)&v44, 0);
        sub_969240(&v46);
        sub_969240(&v44);
        if ( (_BYTE)v57 )
        {
          LOBYTE(v57) = 0;
          sub_969240(&v55);
          sub_969240((__int64 *)v18);
        }
      }
      else if ( (unsigned __int8)sub_B2F0A0(v16, 1) )
      {
        v33 = (unsigned __int8 *)sub_AD6530(*(_QWORD *)(v16 + 8), 1);
        v48 = 0;
        sub_2A62A00((__int64)&v48, v33);
      }
      else
      {
        v48 = 6;
      }
      sub_22EACA0((__int64)v18, v42, (unsigned __int8 *)&v48);
      sub_2A689D0(a1, v16, (unsigned __int8 *)v18, v29);
      sub_22C0090((unsigned __int8 *)v18);
      sub_22C0090((unsigned __int8 *)&v48);
      goto LABEL_40;
    }
    v41 = *(_DWORD *)(v19 + 12);
    if ( !v41 )
      goto LABEL_40;
    v20 = v18;
    v21 = 0;
    WORD1(v22) = HIWORD(v6);
    v23 = (__int64)v20;
    while ( 1 )
    {
      v25 = sub_2A6A1C0(a1, *v43, v21);
      v26 = *v25;
      v48 = *v25;
      if ( v26 > 3u )
        break;
      if ( v26 > 1u )
      {
        v27 = *((_QWORD *)v25 + 1);
        LOWORD(v22) = 256;
        LOWORD(v53) = v26;
        v49 = v27;
        v54 = v27;
      }
      else
      {
        LOWORD(v22) = 256;
        LOWORD(v53) = v26;
      }
      v22 = ((unsigned __int64)(unsigned int)qword_500BEC8 << 32) | (unsigned int)v22;
LABEL_24:
      v24 = sub_2A6A1C0(a1, (unsigned __int8 *)v16, v21);
      sub_2A639B0(a1, v24, v16, v23, v22);
      if ( (unsigned int)(unsigned __int8)v53 - 4 <= 1 )
      {
        if ( v57 > 0x40 && v56 )
          j_j___libc_free_0_0(v56);
        if ( (unsigned int)v55 > 0x40 && v54 )
          j_j___libc_free_0_0(v54);
      }
      if ( (unsigned int)(unsigned __int8)v48 - 4 <= 1 )
      {
        if ( v52 > 0x40 && v51 )
          j_j___libc_free_0_0(v51);
        if ( v50 > 0x40 )
        {
          if ( v49 )
            j_j___libc_free_0_0(v49);
        }
      }
      if ( ++v21 == v41 )
      {
        v18 = (const void **)v23;
        HIWORD(v6) = WORD1(v22);
        goto LABEL_40;
      }
    }
    if ( (unsigned __int8)(v26 - 4) > 1u )
    {
      LOWORD(v22) = 256;
      LOWORD(v53) = v26;
      v22 = ((unsigned __int64)(unsigned int)qword_500BEC8 << 32) | (unsigned int)v22;
LABEL_33:
      if ( (unsigned __int8)(v26 - 4) <= 1u )
      {
        LODWORD(v55) = v50;
        if ( v50 > 0x40 )
          sub_C43780((__int64)&v54, (const void **)&v49);
        else
          v54 = v49;
        v57 = v52;
        if ( v52 > 0x40 )
          sub_C43780((__int64)&v56, (const void **)&v51);
        else
          v56 = v51;
        BYTE1(v53) = HIBYTE(v48);
      }
      goto LABEL_24;
    }
    v50 = *((_DWORD *)v25 + 4);
    if ( v50 > 0x40 )
    {
      v37 = v25;
      sub_C43780((__int64)&v49, (const void **)v25 + 1);
      v25 = v37;
      v52 = *((_DWORD *)v37 + 8);
      if ( v52 <= 0x40 )
        goto LABEL_31;
    }
    else
    {
      v49 = *((_QWORD *)v25 + 1);
      v52 = *((_DWORD *)v25 + 8);
      if ( v52 <= 0x40 )
      {
LABEL_31:
        v51 = *((_QWORD *)v25 + 3);
        goto LABEL_32;
      }
    }
    v38 = v25;
    sub_C43780((__int64)&v51, (const void **)v25 + 3);
    v25 = v38;
LABEL_32:
    LOWORD(v22) = 256;
    HIBYTE(v48) = v25[1];
    v26 = v48;
    LOWORD(v53) = (unsigned __int8)v48;
    v22 = ((unsigned __int64)(unsigned int)qword_500BEC8 << 32) | (unsigned int)v22;
    if ( (unsigned __int8)v48 <= 3u )
    {
      if ( (unsigned __int8)v48 > 1u )
        v54 = v49;
      goto LABEL_24;
    }
    goto LABEL_33;
  }
}
