// Function: sub_25157B0
// Address: 0x25157b0
//
__int64 __fastcall sub_25157B0(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 (__fastcall *a5)(__int64, __int64, __int64, _QWORD *, __int64 **),
        __int64 a6)
{
  unsigned int v10; // r12d
  unsigned __int8 *v12; // rax
  int v13; // edx
  unsigned __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // rdi
  unsigned __int8 **v19; // rcx
  unsigned __int8 *v20; // r9
  unsigned __int64 v21; // rdi
  char v22; // al
  __int64 v23; // rax
  __int64 v24; // r15
  int v25; // ebx
  __int64 v26; // r12
  __int64 v27; // r13
  unsigned int v28; // esi
  __int64 v29; // rcx
  __int64 v30; // rdi
  __int64 *v31; // r10
  int v32; // r9d
  unsigned int v33; // edx
  _QWORD *v34; // rax
  unsigned __int8 *v35; // r11
  unsigned __int64 *v36; // rax
  _QWORD *v37; // rdi
  int v38; // ecx
  int v39; // r10d
  int v40; // eax
  int v41; // edx
  __int64 v42; // rbx
  __int64 *v43; // [rsp+10h] [rbp-120h]
  int v44; // [rsp+1Ch] [rbp-114h]
  __int64 v46; // [rsp+28h] [rbp-108h]
  __int64 v47; // [rsp+30h] [rbp-100h]
  unsigned __int64 v49; // [rsp+48h] [rbp-E8h] BYREF
  unsigned __int8 *v50; // [rsp+50h] [rbp-E0h] BYREF
  __int64 *v51; // [rsp+58h] [rbp-D8h] BYREF
  _QWORD v52[3]; // [rsp+60h] [rbp-D0h] BYREF
  int v53; // [rsp+78h] [rbp-B8h] BYREF
  _QWORD *v54; // [rsp+80h] [rbp-B0h]
  int *v55; // [rsp+88h] [rbp-A8h]
  int *v56; // [rsp+90h] [rbp-A0h]
  __int64 v57; // [rsp+98h] [rbp-98h]
  __int64 *v58; // [rsp+A0h] [rbp-90h] BYREF
  _BYTE *v59; // [rsp+A8h] [rbp-88h]
  __int64 v60; // [rsp+B0h] [rbp-80h]
  _BYTE v61[120]; // [rsp+B8h] [rbp-78h] BYREF

  if ( !a4 )
    return 1;
  v10 = 1;
  if ( (unsigned int)(char)sub_2509800(a2) > 1 )
  {
    v49 = 0;
    v12 = (unsigned __int8 *)(*a2 & 0xFFFFFFFFFFFFFFFCLL);
    if ( (*a2 & 3) == 3 )
      v12 = (unsigned __int8 *)*((_QWORD *)v12 + 3);
    v13 = *v12;
    if ( (unsigned __int8)v13 <= 0x1Cu
      || (v14 = (unsigned int)(v13 - 34), (unsigned __int8)v14 > 0x33u)
      || (v15 = 0x8000000000041LL, !_bittest64(&v15, v14)) )
    {
      v12 = sub_250CBE0(a2, (__int64)a2);
    }
    v16 = a1;
    v50 = v12;
    v17 = *(unsigned int *)(a1 + 24);
    v18 = *(_QWORD *)(a1 + 8);
    if ( (_DWORD)v17 )
    {
      v16 = ((_DWORD)v17 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v19 = (unsigned __int8 **)(v18 + 16 * v16);
      v20 = *v19;
      if ( v12 == *v19 )
      {
LABEL_11:
        if ( v19 != (unsigned __int8 **)(v18 + 16 * v17) )
        {
          v49 = (unsigned __int64)v19[1];
          goto LABEL_13;
        }
      }
      else
      {
        v38 = 1;
        while ( v20 != (unsigned __int8 *)-4096LL )
        {
          v39 = v38 + 1;
          v16 = ((_DWORD)v17 - 1) & (unsigned int)(v38 + v16);
          v19 = (unsigned __int8 **)(v18 + 16LL * (unsigned int)v16);
          v20 = *v19;
          if ( v12 == *v19 )
            goto LABEL_11;
          v38 = v39;
        }
      }
    }
    v49 = sub_250CD80(a2, v16);
LABEL_13:
    v21 = *a2 & 0xFFFFFFFFFFFFFFFCLL;
    if ( (*a2 & 3) == 3 )
      v21 = *(_QWORD *)(v21 + 24);
    v43 = (__int64 *)sub_BD5C60(v21);
    v22 = sub_2509800(a2);
    if ( v22 == 6 )
    {
      v44 = sub_250CB50(a2, 1) + 1;
    }
    else if ( v22 > 6 )
    {
      if ( v22 != 7 )
        goto LABEL_36;
      v44 = sub_250CB50(a2, 0) + 1;
    }
    else
    {
      if ( v22 > 3 )
      {
        v44 = -1;
        goto LABEL_19;
      }
      if ( v22 <= 1 )
LABEL_36:
        BUG();
      v44 = 0;
    }
LABEL_19:
    v23 = sub_A74490(&v49, v44);
    v52[0] = 0;
    v46 = v23;
    v55 = &v53;
    v56 = &v53;
    v52[1] = 0;
    v58 = v43;
    v59 = v61;
    v60 = 0x800000000LL;
    v53 = 0;
    v54 = 0;
    v57 = 0;
    v47 = a3 + 4 * a4;
    if ( a3 == v47 )
    {
      v37 = 0;
      v10 = 1;
LABEL_32:
      sub_2507480(v37);
      return v10;
    }
    v24 = a3;
    v25 = 1;
    v26 = a6;
    v27 = v24;
    do
    {
      if ( a5(v26, v27, v46, v52, &v58) )
        v25 = 0;
      v27 += 4;
    }
    while ( v47 != v27 );
    v10 = v25;
    if ( v25 == 1 )
    {
LABEL_29:
      if ( v59 != v61 )
        _libc_free((unsigned __int64)v59);
      v37 = v54;
      goto LABEL_32;
    }
    v49 = sub_A7A440((__int64 *)&v49, v43, v44, (__int64)v52);
    v49 = sub_A7B2C0((__int64 *)&v49, v43, v44, (__int64)&v58);
    v28 = *(_DWORD *)(a1 + 24);
    if ( v28 )
    {
      v29 = (__int64)v50;
      v30 = *(_QWORD *)(a1 + 8);
      v31 = 0;
      v32 = 1;
      v33 = (v28 - 1) & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
      v34 = (_QWORD *)(v30 + 16LL * v33);
      v35 = (unsigned __int8 *)*v34;
      if ( (unsigned __int8 *)*v34 == v50 )
      {
LABEL_27:
        v36 = v34 + 1;
LABEL_28:
        *v36 = v49;
        goto LABEL_29;
      }
      while ( v35 != (unsigned __int8 *)-4096LL )
      {
        if ( !v31 && v35 == (unsigned __int8 *)-8192LL )
          v31 = v34;
        v33 = (v28 - 1) & (v32 + v33);
        v34 = (_QWORD *)(v30 + 16LL * v33);
        v35 = (unsigned __int8 *)*v34;
        if ( v50 == (unsigned __int8 *)*v34 )
          goto LABEL_27;
        ++v32;
      }
      if ( !v31 )
        v31 = v34;
      ++*(_QWORD *)a1;
      v40 = *(_DWORD *)(a1 + 16);
      v51 = v31;
      v41 = v40 + 1;
      if ( 4 * (v40 + 1) < 3 * v28 )
      {
        v42 = a1;
        if ( v28 - *(_DWORD *)(a1 + 20) - v41 > v28 >> 3 )
        {
LABEL_57:
          *(_DWORD *)(a1 + 16) = v41;
          if ( *v31 != -4096 )
            --*(_DWORD *)(a1 + 20);
          *v31 = v29;
          v36 = (unsigned __int64 *)(v31 + 1);
          v31[1] = 0;
          goto LABEL_28;
        }
LABEL_62:
        sub_2515060(v42, v28);
        sub_2510B30(v42, (__int64 *)&v50, &v51);
        v29 = (__int64)v50;
        v31 = v51;
        v41 = *(_DWORD *)(v42 + 16) + 1;
        goto LABEL_57;
      }
    }
    else
    {
      v51 = 0;
      ++*(_QWORD *)a1;
    }
    v42 = a1;
    v28 *= 2;
    goto LABEL_62;
  }
  return v10;
}
