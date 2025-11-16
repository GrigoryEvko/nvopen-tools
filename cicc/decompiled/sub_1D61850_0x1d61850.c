// Function: sub_1D61850
// Address: 0x1d61850
//
__int64 __fastcall sub_1D61850(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r15
  __int64 v9; // rdx
  char v10; // bl
  __int64 (__fastcall *v11)(__int64, __int64, __m128, double, double, double, double, double, double, __m128, __int64, int *, __int64, __int64, _QWORD *); // r11
  __int64 v12; // rax
  _QWORD *v13; // rdi
  __int64 v14; // rdx
  signed __int64 v15; // r8
  bool v16; // al
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rax
  _BYTE *v20; // r15
  char v21; // r13
  unsigned __int64 v22; // rbx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r14
  __int64 v26; // rcx
  _QWORD *v27; // rax
  char v28; // dl
  __int64 v29; // r13
  __int64 v30; // rbx
  __int64 v31; // rsi
  __int64 v32; // rax
  unsigned int v33; // edx
  __int64 v34; // rax
  unsigned int v35; // edx
  __int64 (*v36)(); // rax
  __int64 v38; // [rsp+10h] [rbp-100h]
  __int64 v39; // [rsp+18h] [rbp-F8h]
  unsigned int v40; // [rsp+24h] [rbp-ECh]
  __int64 v41; // [rsp+28h] [rbp-E8h]
  __int64 v42; // [rsp+30h] [rbp-E0h]
  __int64 v43; // [rsp+40h] [rbp-D0h]
  __int64 v44; // [rsp+48h] [rbp-C8h]
  __int64 (__fastcall *v45)(_QWORD *, __int64, __int64, unsigned int *, unsigned __int64 *, _QWORD, _QWORD); // [rsp+58h] [rbp-B8h]
  signed __int64 v46; // [rsp+58h] [rbp-B8h]
  unsigned __int64 v47; // [rsp+58h] [rbp-B8h]
  __int64 v48; // [rsp+60h] [rbp-B0h]
  __int64 v49; // [rsp+68h] [rbp-A8h]
  int v50; // [rsp+70h] [rbp-A0h]
  char v51; // [rsp+74h] [rbp-9Ch]
  unsigned __int8 v52; // [rsp+75h] [rbp-9Bh]
  unsigned __int8 v53; // [rsp+76h] [rbp-9Ah]
  char v54; // [rsp+76h] [rbp-9Ah]
  char v55; // [rsp+77h] [rbp-99h]
  __int64 *v56; // [rsp+78h] [rbp-98h]
  unsigned int v57; // [rsp+84h] [rbp-8Ch] BYREF
  _QWORD *v58; // [rsp+88h] [rbp-88h] BYREF
  _BYTE *v59; // [rsp+90h] [rbp-80h] BYREF
  __int64 v60; // [rsp+98h] [rbp-78h]
  _BYTE v61[16]; // [rsp+A0h] [rbp-70h] BYREF
  unsigned __int64 v62[2]; // [rsp+B0h] [rbp-60h] BYREF
  _BYTE v63[80]; // [rsp+C0h] [rbp-50h] BYREF

  v50 = a5;
  v44 = *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8);
  if ( *(_QWORD *)a3 != v44 )
  {
    v56 = *(__int64 **)a3;
    v7 = a1;
    v49 = a1 + 488;
    v52 = 0;
    v43 = a1 + 320;
    while ( 1 )
    {
      v18 = *v56;
      v58 = (_QWORD *)v18;
      if ( (*(_BYTE *)(v18 + 23) & 0x40) != 0 )
      {
        if ( *(_BYTE *)(**(_QWORD **)(v18 - 8) + 16LL) == 54 )
          goto LABEL_22;
      }
      else if ( *(_BYTE *)(*(_QWORD *)(v18 - 24LL * (*(_DWORD *)(v18 + 20) & 0xFFFFFFF)) + 16LL) == 54 )
      {
LABEL_22:
        v19 = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)v19 >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, (const void *)(a4 + 16), 0, 8, a5, a6);
          v19 = *(unsigned int *)(a4 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * v19) = v58;
        ++*(_DWORD *)(a4 + 8);
        goto LABEL_19;
      }
      v9 = *(_QWORD *)(v7 + 176);
      if ( !v9 )
        return 0;
      v55 = *(_BYTE *)(v9 + 81537);
      if ( !v55 )
        return 0;
      v10 = byte_4FC2CE0;
      if ( byte_4FC2CE0 )
        return 0;
      v11 = sub_1D61430((__int64 *)v18, v43, v9, v49);
      if ( !v11 )
      {
        sub_14EF3D0(a4, &v58);
        goto LABEL_19;
      }
      v48 = 0;
      v12 = *(unsigned int *)(a2 + 8);
      if ( (_DWORD)v12 )
        v48 = *(_QWORD *)(*(_QWORD *)a2 + 8 * v12 - 8);
      v13 = *(_QWORD **)(v7 + 176);
      v62[0] = (unsigned __int64)v63;
      v45 = (__int64 (__fastcall *)(_QWORD *, __int64, __int64, unsigned int *, unsigned __int64 *, _QWORD, _QWORD))v11;
      v62[1] = 0x400000000LL;
      v57 = 0;
      v53 = sub_1D5EF60(v13, v58) ^ 1;
      v14 = v45(v58, a2, v49, &v57, v62, 0, *(_QWORD *)(v7 + 176));
      v15 = v57 + v50 - (unsigned __int64)v53;
      if ( v15 > 0 )
      {
        if ( !byte_4FC2C00 )
        {
          if ( v15 != 1 )
          {
LABEL_14:
            sub_1D5ABA0((__int64 *)a2, v48);
            v17 = *(unsigned int *)(a4 + 8);
            if ( (unsigned int)v17 >= *(_DWORD *)(a4 + 12) )
            {
              sub_16CD150(a4, (const void *)(a4 + 16), 0, 8, a5, a6);
              v17 = *(unsigned int *)(a4 + 8);
            }
            *(_QWORD *)(*(_QWORD *)a4 + 8 * v17) = v58;
            ++*(_DWORD *)(a4 + 8);
            goto LABEL_17;
          }
          goto LABEL_13;
        }
      }
      else
      {
        if ( !byte_4FC2C00 )
        {
          v15 = 0;
LABEL_13:
          v46 = v15;
          v16 = sub_1D5E480(*(_QWORD *)(v7 + 176), *(_QWORD *)(v7 + 904), v14);
          v15 = v46;
          if ( !v16 )
            goto LABEL_14;
          goto LABEL_26;
        }
        v15 = 0;
      }
LABEL_26:
      v59 = v61;
      v60 = 0x200000000LL;
      sub_1D61850(v7, a2, v62, &v59, v15);
      LODWORD(a5) = (_DWORD)v59;
      v40 = v53;
      if ( v59 == &v59[8 * (unsigned int)v60] )
        goto LABEL_67;
      v42 = v7;
      v20 = &v59[8 * (unsigned int)v60];
      v21 = v10;
      v22 = (unsigned __int64)v59;
      v41 = a2;
      do
      {
        while ( 1 )
        {
          v25 = *(_QWORD *)v22;
          if ( (*(_BYTE *)(*(_QWORD *)v22 + 23LL) & 0x40) != 0 )
          {
            v23 = **(_QWORD **)(v25 - 8);
            if ( *(_BYTE *)(v23 + 16) != 54 )
              break;
          }
          else
          {
            v23 = *(_QWORD *)(v25 - 24LL * (*(_DWORD *)(v25 + 20) & 0xFFFFFFF));
            if ( *(_BYTE *)(v23 + 16) != 54 )
              break;
          }
          if ( byte_4FC2C00 )
            break;
          if ( v57 <= v40 )
            break;
          v26 = *(_QWORD *)(v23 + 8);
          if ( !v26 || !*(_QWORD *)(v26 + 8) )
            break;
          v38 = *(_QWORD *)(v23 + 8);
          v39 = *(_QWORD *)(v42 + 176);
          v27 = sub_1648700(v38);
          v51 = v21;
          a6 = *v27;
          v47 = v22;
          v54 = *((_BYTE *)v27 + 16);
          v28 = v54;
          v29 = *v27;
          v30 = v38;
          if ( v54 != 62 )
          {
LABEL_39:
            if ( v28 != 61 )
              goto LABEL_48;
            v31 = *v27;
            if ( v29 != *v27 )
            {
              v32 = v29;
              if ( *(_BYTE *)(v29 + 8) == 16 )
                v32 = **(_QWORD **)(v29 + 16);
              v33 = *(_DWORD *)(v32 + 8);
              v34 = v31;
              v35 = v33 >> 8;
              if ( *(_BYTE *)(v31 + 8) == 16 )
                v34 = **(_QWORD **)(v31 + 16);
              a5 = v29;
              if ( v35 <= *(_DWORD *)(v34 + 8) >> 8 )
              {
                a5 = v31;
                v31 = v29;
              }
              v36 = *(__int64 (**)())(*(_QWORD *)v39 + 816LL);
              if ( v36 == sub_1D5A400 || !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64))v36)(v39, v31, a5) )
                goto LABEL_48;
            }
            goto LABEL_61;
          }
          while ( v28 == 62 && v29 == *v27 )
          {
LABEL_61:
            v30 = *(_QWORD *)(v30 + 8);
            if ( !v30 )
            {
              v22 = v47;
              goto LABEL_29;
            }
            v27 = sub_1648700(v30);
            v28 = *((_BYTE *)v27 + 16);
            if ( v54 != 62 )
              goto LABEL_39;
          }
LABEL_48:
          v21 = v51;
          v22 = v47 + 8;
          if ( v20 == (_BYTE *)(v47 + 8) )
            goto LABEL_49;
        }
LABEL_29:
        v24 = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)v24 >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, (const void *)(a4 + 16), 0, 8, a5, a6);
          v24 = *(unsigned int *)(a4 + 8);
        }
        v22 += 8LL;
        v21 = v55;
        *(_QWORD *)(*(_QWORD *)a4 + 8 * v24) = v25;
        ++*(_DWORD *)(a4 + 8);
      }
      while ( v20 != (_BYTE *)v22 );
LABEL_49:
      v7 = v42;
      a2 = v41;
      if ( v21 )
      {
        if ( v59 != v61 )
          _libc_free((unsigned __int64)v59);
        if ( (_BYTE *)v62[0] != v63 )
          _libc_free(v62[0]);
        v52 = v21;
        goto LABEL_19;
      }
LABEL_67:
      sub_1D5ABA0((__int64 *)a2, v48);
      sub_14EF3D0(a4, &v58);
      if ( v59 != v61 )
        _libc_free((unsigned __int64)v59);
LABEL_17:
      if ( (_BYTE *)v62[0] != v63 )
        _libc_free(v62[0]);
LABEL_19:
      if ( (__int64 *)v44 == ++v56 )
        return v52;
    }
  }
  return 0;
}
