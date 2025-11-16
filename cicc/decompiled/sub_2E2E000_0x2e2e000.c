// Function: sub_2E2E000
// Address: 0x2e2e000
//
void __fastcall sub_2E2E000(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 (*v3)(); // rax
  _QWORD *v5; // r12
  _DWORD *v6; // rax
  unsigned int v7; // r14d
  bool v8; // zf
  _DWORD *v9; // rbx
  __int64 v10; // rcx
  int v11; // eax
  _QWORD *v12; // rsi
  int v13; // r12d
  signed int v14; // r15d
  __int64 v15; // rdi
  _DWORD *v16; // rax
  _DWORD *v17; // rcx
  __int64 (*v18)(); // rax
  unsigned int v19; // edx
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rax
  int *v22; // r8
  __int64 (*v23)(); // rax
  __int64 v24; // r9
  __int64 v25; // rsi
  int v26; // eax
  __int64 v27; // rcx
  unsigned int *v28; // rdx
  unsigned int v29; // r15d
  int v30; // r12d
  __int64 v31; // rax
  __int64 (*v32)(); // r8
  unsigned __int8 v33; // al
  bool v34; // cc
  char v35; // [rsp+44h] [rbp-1C4h]
  unsigned __int8 v36; // [rsp+53h] [rbp-1B5h] BYREF
  unsigned int v37; // [rsp+54h] [rbp-1B4h] BYREF
  __int64 v38; // [rsp+58h] [rbp-1B0h] BYREF
  __int64 v39; // [rsp+60h] [rbp-1A8h] BYREF
  __int64 v40; // [rsp+68h] [rbp-1A0h]
  __int64 v41; // [rsp+70h] [rbp-198h]
  __int64 v42; // [rsp+78h] [rbp-190h]
  _BYTE *v43; // [rsp+80h] [rbp-188h]
  __int64 v44; // [rsp+88h] [rbp-180h]
  _BYTE v45[32]; // [rsp+90h] [rbp-178h] BYREF
  __int64 v46; // [rsp+B0h] [rbp-158h] BYREF
  __int64 v47; // [rsp+B8h] [rbp-150h]
  __int64 v48; // [rsp+C0h] [rbp-148h]
  __int64 v49; // [rsp+C8h] [rbp-140h]
  _BYTE *v50; // [rsp+D0h] [rbp-138h]
  __int64 v51; // [rsp+D8h] [rbp-130h]
  _BYTE v52[32]; // [rsp+E0h] [rbp-128h] BYREF
  __int64 v53; // [rsp+100h] [rbp-108h] BYREF
  __int64 v54; // [rsp+108h] [rbp-100h]
  __int64 v55; // [rsp+110h] [rbp-F8h]
  __int64 v56; // [rsp+118h] [rbp-F0h]
  _BYTE *v57; // [rsp+120h] [rbp-E8h]
  __int64 v58; // [rsp+128h] [rbp-E0h]
  _BYTE v59[32]; // [rsp+130h] [rbp-D8h] BYREF
  _BYTE *v60; // [rsp+150h] [rbp-B8h] BYREF
  __int64 v61; // [rsp+158h] [rbp-B0h]
  _BYTE v62[72]; // [rsp+160h] [rbp-A8h] BYREF
  int v63; // [rsp+1A8h] [rbp-60h] BYREF
  unsigned __int64 v64; // [rsp+1B0h] [rbp-58h]
  int *v65; // [rsp+1B8h] [rbp-50h]
  int *v66; // [rsp+1C0h] [rbp-48h]
  __int64 v67; // [rsp+1C8h] [rbp-40h]

  v3 = *(__int64 (**)())(*(_QWORD *)a2 + 136LL);
  if ( v3 == sub_2DD19D0 )
    BUG();
  v5 = a1;
  v6 = (_DWORD *)((__int64 (__fastcall *)(__int64))v3)(a2);
  v7 = *(_DWORD *)(a3 + 68);
  v36 = 0;
  v8 = v6[2] == 1;
  v9 = v6;
  v38 = 0;
  v60 = v62;
  v61 = 0x1000000000LL;
  v63 = 0;
  v64 = 0;
  v65 = &v63;
  v66 = &v63;
  v67 = 0;
  v35 = v8;
  if ( v7 != -1 )
  {
    v42 = 0;
    v43 = v45;
    v44 = 0x800000000LL;
    v49 = 0;
    v50 = v52;
    v51 = 0x800000000LL;
    v56 = 0;
    v57 = v59;
    v58 = 0x800000000LL;
    v39 = 0;
    v40 = 0;
    v41 = 0;
    v46 = 0;
    v47 = 0;
    v48 = 0;
    v53 = 0;
    v54 = 0;
    v55 = 0;
    v23 = *(__int64 (**)())(*(_QWORD *)v6 + 24LL);
    if ( v23 == sub_2E2CB60
      || ((unsigned __int8 (__fastcall *)(_DWORD *, _QWORD))v23)(
           v9,
           *(unsigned __int8 *)(*(_QWORD *)(a3 + 8) + 40LL * (*(_DWORD *)(a3 + 32) + v7) + 20)) )
    {
      sub_2E2CB70(a1, a3, v7, &v38, v35 & 1, &v36);
    }
    v25 = *(_QWORD *)(a3 + 8);
    v26 = *(_DWORD *)(a3 + 32);
    v27 = -858993459 * (unsigned int)((*(_QWORD *)(a3 + 16) - v25) >> 3) - v26;
    if ( !(_DWORD)v27 )
    {
LABEL_46:
      sub_2E2D830(v5, (__int64)&v39, (__int64)&v60, a3, v35, &v38, &v36);
      sub_2E2D830(v5, (__int64)&v46, (__int64)&v60, a3, v35, &v38, &v36);
      sub_2E2D830(v5, (__int64)&v53, (__int64)&v60, a3, v35, &v38, &v36);
      if ( v57 != v59 )
        _libc_free((unsigned __int64)v57);
      sub_C7D6A0(v54, 4LL * (unsigned int)v56, 4);
      if ( v50 != v52 )
        _libc_free((unsigned __int64)v50);
      sub_C7D6A0(v47, 4LL * (unsigned int)v49, 4);
      if ( v43 != v45 )
        _libc_free((unsigned __int64)v43);
      sub_C7D6A0(v40, 4LL * (unsigned int)v42, 4);
      goto LABEL_3;
    }
    v28 = &v37;
    v29 = 0;
    v30 = -858993459 * ((*(_QWORD *)(a3 + 16) - v25) >> 3) - v26;
    while ( 1 )
    {
      v31 = v25 + 40LL * (v29 + v26);
      if ( *(_QWORD *)(v31 + 8) == -1 || v7 == v29 )
        goto LABEL_36;
      v32 = *(__int64 (**)())(*(_QWORD *)v9 + 24LL);
      if ( v32 == sub_2E2CB60 )
      {
        v33 = *(_BYTE *)(v31 + 36);
        v34 = v33 <= 2u;
        if ( v33 == 2 )
          goto LABEL_55;
      }
      else
      {
        if ( !((unsigned __int8 (__fastcall *)(_DWORD *, _QWORD))v32)(v9, *(unsigned __int8 *)(v31 + 20)) )
          goto LABEL_36;
        v33 = *(_BYTE *)(*(_QWORD *)(a3 + 8) + 40LL * (*(_DWORD *)(a3 + 32) + v29) + 36);
        v34 = v33 <= 2u;
        if ( v33 == 2 )
        {
LABEL_55:
          v37 = v29;
          sub_2E2DA00((__int64)&v46, &v37, (__int64)v28, v27, (__int64)v32, v24);
          goto LABEL_36;
        }
      }
      if ( v34 )
      {
        if ( v33 )
        {
          v37 = v29;
          sub_2E2DA00((__int64)&v39, &v37, (__int64)v28, v27, (__int64)v32, v24);
        }
LABEL_36:
        if ( v30 == ++v29 )
          goto LABEL_45;
        goto LABEL_37;
      }
      if ( v33 != 3 )
        BUG();
      v37 = v29++;
      sub_2E2DA00((__int64)&v53, &v37, (__int64)v28, v27, (__int64)v32, v24);
      if ( v30 == v29 )
      {
LABEL_45:
        v5 = a1;
        goto LABEL_46;
      }
LABEL_37:
      v26 = *(_DWORD *)(a3 + 32);
      v25 = *(_QWORD *)(a3 + 8);
    }
  }
LABEL_3:
  v10 = *(_QWORD *)(a3 + 8);
  v11 = *(_DWORD *)(a3 + 32);
  if ( -858993459 * (unsigned int)((*(_QWORD *)(a3 + 16) - v10) >> 3) != v11 )
  {
    v12 = v5;
    v13 = -858993459 * ((*(_QWORD *)(a3 + 16) - v10) >> 3) - v11;
    v14 = 0;
    while ( 1 )
    {
      v15 = v10 + 40LL * (unsigned int)(v14 + v11);
      if ( *(_QWORD *)(v15 + 8) == -1 || v14 == *(_DWORD *)(a3 + 68) )
        break;
      if ( v67 )
      {
        v21 = v64;
        if ( v64 )
        {
          v22 = &v63;
          do
          {
            if ( v14 > *(_DWORD *)(v21 + 32) )
            {
              v21 = *(_QWORD *)(v21 + 24);
            }
            else
            {
              v22 = (int *)v21;
              v21 = *(_QWORD *)(v21 + 16);
            }
          }
          while ( v21 );
          if ( v22 != &v63 && v14 >= v22[8] )
            break;
        }
      }
      else
      {
        v16 = v60;
        v17 = &v60[4 * (unsigned int)v61];
        if ( v60 != (_BYTE *)v17 )
        {
          while ( v14 != *v16 )
          {
            if ( v17 == ++v16 )
              goto LABEL_17;
          }
          if ( v17 != v16 )
            break;
        }
      }
LABEL_17:
      v18 = *(__int64 (**)())(*(_QWORD *)v9 + 24LL);
      if ( v18 != sub_2E2CB60
        && !((unsigned __int8 (__fastcall *)(_DWORD *, _QWORD))v18)(v9, *(unsigned __int8 *)(v15 + 20)) )
      {
        break;
      }
      v19 = v14++;
      sub_2E2CB70(v12, a3, v19, &v38, v35 & 1, &v36);
      if ( v13 == v14 )
        goto LABEL_19;
LABEL_14:
      v11 = *(_DWORD *)(a3 + 32);
      v10 = *(_QWORD *)(a3 + 8);
    }
    if ( v13 == ++v14 )
      goto LABEL_19;
    goto LABEL_14;
  }
LABEL_19:
  v20 = v64;
  *(_QWORD *)(a3 + 656) = v38;
  *(_BYTE *)(a3 + 664) = v36;
  sub_2E2D4A0(v20);
  if ( v60 != v62 )
    _libc_free((unsigned __int64)v60);
}
