// Function: sub_2D61E30
// Address: 0x2d61e30
//
__int64 __fastcall sub_2D61E30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r14
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 (__fastcall *v11)(int, int, int, int, int, int, __int64); // r11
  __int64 v12; // rax
  __int64 *v13; // rdi
  int v14; // r12d
  __int64 **v15; // rdx
  signed __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rax
  __int64 v19; // r13
  __int64 v20; // r9
  unsigned __int64 v21; // rbx
  _BYTE *v22; // r10
  unsigned __int8 v23; // cl
  _QWORD *v24; // rdx
  _BYTE *v25; // rax
  __int64 v26; // rax
  __int64 v27; // r12
  _QWORD *v28; // rdx
  __int64 v29; // r14
  __int64 v30; // rdi
  unsigned __int8 *v31; // rax
  __int64 v32; // r13
  char v33; // dl
  char v34; // r12
  __int64 v35; // rsi
  __int64 v36; // rdx
  unsigned int v37; // r9d
  __int64 v38; // rdx
  unsigned int v39; // eax
  __int64 v40; // rdx
  __int64 (*v41)(); // rax
  char v42; // al
  __int64 v44; // [rsp+0h] [rbp-110h]
  _BYTE *v45; // [rsp+8h] [rbp-108h]
  unsigned int v46; // [rsp+14h] [rbp-FCh]
  __int64 v47; // [rsp+18h] [rbp-F8h]
  unsigned __int8 v48; // [rsp+2Ah] [rbp-E6h]
  unsigned __int8 v49; // [rsp+2Bh] [rbp-E5h]
  int v50; // [rsp+2Ch] [rbp-E4h]
  __int64 v51; // [rsp+30h] [rbp-E0h]
  unsigned __int8 v52; // [rsp+38h] [rbp-D8h]
  unsigned __int8 v53; // [rsp+38h] [rbp-D8h]
  _BYTE *v54; // [rsp+38h] [rbp-D8h]
  __int64 v55; // [rsp+40h] [rbp-D0h]
  __int64 (__fastcall *v56)(int, int, int, int, int, int, __int64); // [rsp+48h] [rbp-C8h]
  signed __int64 v57; // [rsp+48h] [rbp-C8h]
  __int64 v58; // [rsp+48h] [rbp-C8h]
  unsigned __int8 v59; // [rsp+58h] [rbp-B8h]
  unsigned __int8 v60; // [rsp+58h] [rbp-B8h]
  __int64 v61; // [rsp+60h] [rbp-B0h]
  __int64 v62; // [rsp+68h] [rbp-A8h]
  __int64 *v64; // [rsp+78h] [rbp-98h]
  unsigned int v65; // [rsp+8Ch] [rbp-84h] BYREF
  _BYTE *v66; // [rsp+90h] [rbp-80h] BYREF
  __int64 v67; // [rsp+98h] [rbp-78h]
  _BYTE v68[16]; // [rsp+A0h] [rbp-70h] BYREF
  _BYTE *v69; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v70; // [rsp+B8h] [rbp-58h]
  _BYTE v71[80]; // [rsp+C0h] [rbp-50h] BYREF

  v5 = a1;
  v50 = a5;
  v55 = *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8);
  if ( *(_QWORD *)a3 != v55 )
  {
    v64 = *(__int64 **)a3;
    v61 = a1 + 344;
    v49 = 0;
    v51 = a1 + 184;
    while ( 1 )
    {
      v19 = *v64;
      if ( (*(_BYTE *)(*v64 + 7) & 0x40) != 0 )
      {
        if ( ***(_BYTE ***)(v19 - 8) == 61 )
          goto LABEL_23;
      }
      else if ( **(_BYTE **)(v19 - 32LL * (*(_DWORD *)(v19 + 4) & 0x7FFFFFF)) == 61 )
      {
LABEL_23:
        sub_9C95B0(a4, v19);
        goto LABEL_20;
      }
      v7 = *(_QWORD *)(v5 + 16);
      v52 = *(_BYTE *)(v7 + 537005);
      if ( !v52 || byte_5017D48 )
        return 0;
      v11 = sub_2D5AD50((char *)v19, v51, v7, v61, a5);
      if ( !v11 )
        goto LABEL_23;
      v62 = 0;
      v12 = *(unsigned int *)(a2 + 8);
      if ( (_DWORD)v12 )
      {
        v8 = *(_QWORD *)a2;
        v62 = *(_QWORD *)(*(_QWORD *)a2 + 8 * v12 - 8);
      }
      v13 = *(__int64 **)(v5 + 16);
      v69 = v71;
      v56 = v11;
      v70 = 0x400000000LL;
      v65 = 0;
      v14 = sub_2D5C100(v13, (unsigned __int8 *)v19, v8, v9, v10) ^ 1;
      v15 = (__int64 **)v56(v19, a2, v61, (int)&v65, (int)&v69, 0, *(_QWORD *)(v5 + 16));
      v16 = v65 + v50 - (unsigned __int64)(unsigned __int8)v14;
      if ( v16 > 0 )
      {
        if ( byte_5017C68 )
          goto LABEL_28;
        if ( v16 != 1 )
          goto LABEL_15;
      }
      else
      {
        if ( byte_5017C68 )
        {
          v16 = 0;
          goto LABEL_28;
        }
        v16 = 0;
      }
      v57 = v16;
      if ( !sub_2D5BD10(*(_QWORD *)(v5 + 16), *(_QWORD *)(v5 + 816), v15)
        || (v16 = v57, !(_BYTE)v14) && (unsigned int)v70 > 1 )
      {
LABEL_15:
        sub_2D57BD0((__int64 *)a2, v62);
        v18 = *(unsigned int *)(a4 + 8);
        if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
        {
          sub_C8D5F0(a4, (const void *)(a4 + 16), v18 + 1, 8u, a5, v17);
          v18 = *(unsigned int *)(a4 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * v18) = v19;
        ++*(_DWORD *)(a4 + 8);
LABEL_18:
        if ( v69 != v71 )
          _libc_free((unsigned __int64)v69);
        goto LABEL_20;
      }
LABEL_28:
      v66 = v68;
      v67 = 0x200000000LL;
      sub_2D61E30(v5, a2, &v69, &v66, v16);
      v21 = (unsigned __int64)v66;
      v22 = &v66[8 * (unsigned int)v67];
      v46 = (unsigned __int8)v14;
      if ( v22 == v66 )
        goto LABEL_65;
      v47 = v19;
      v23 = 0;
      v58 = v5;
      a5 = v52;
      while ( 1 )
      {
LABEL_35:
        v27 = *(_QWORD *)v21;
        if ( (*(_BYTE *)(*(_QWORD *)v21 + 7LL) & 0x40) != 0 )
        {
          v24 = *(_QWORD **)(v27 - 8);
          v25 = (_BYTE *)*v24;
          if ( *(_BYTE *)*v24 != 61 )
            goto LABEL_31;
        }
        else
        {
          v28 = (_QWORD *)(v27 - 32LL * (*(_DWORD *)(v27 + 4) & 0x7FFFFFF));
          v25 = (_BYTE *)*v28;
          if ( *(_BYTE *)*v28 != 61 )
            goto LABEL_31;
        }
        if ( !byte_5017C68 && v65 > v46 )
        {
          v29 = *((_QWORD *)v25 + 2);
          if ( !v29 || *(_QWORD *)(v29 + 8) )
            break;
        }
LABEL_31:
        v26 = *(unsigned int *)(a4 + 8);
        if ( v26 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
        {
          v54 = v22;
          v60 = a5;
          sub_C8D5F0(a4, (const void *)(a4 + 16), v26 + 1, 8u, a5, v20);
          v26 = *(unsigned int *)(a4 + 8);
          v22 = v54;
          a5 = v60;
        }
        v23 = a5;
        *(_QWORD *)(*(_QWORD *)a4 + 8 * v26) = v27;
        ++*(_DWORD *)(a4 + 8);
LABEL_34:
        v21 += 8LL;
        if ( v22 == (_BYTE *)v21 )
          goto LABEL_57;
      }
      v44 = *(_QWORD *)v21;
      v30 = *(_QWORD *)(v58 + 16);
      v31 = *(unsigned __int8 **)(v29 + 24);
      v20 = *v31;
      v32 = *((_QWORD *)v31 + 1);
      v33 = *v31;
      v34 = *v31;
      while ( v34 != 69 )
      {
        if ( v33 != 68 )
          goto LABEL_34;
        v35 = *((_QWORD *)v31 + 1);
        if ( v32 != v35 )
        {
          v36 = v32;
          if ( (unsigned int)*(unsigned __int8 *)(v32 + 8) - 17 <= 1 )
            v36 = **(_QWORD **)(v32 + 16);
          v37 = *(_DWORD *)(v36 + 8);
          v38 = *((_QWORD *)v31 + 1);
          v20 = v37 >> 8;
          if ( (unsigned int)*(unsigned __int8 *)(v35 + 8) - 17 <= 1 )
            v38 = **(_QWORD **)(v35 + 16);
          v39 = *(_DWORD *)(v38 + 8);
          v40 = v32;
          if ( (unsigned int)v20 <= v39 >> 8 )
          {
            v40 = v35;
            v35 = v32;
          }
          v41 = *(__int64 (**)())(*(_QWORD *)v30 + 1424LL);
          if ( v41 == sub_2D56670 )
            goto LABEL_34;
          v45 = v22;
          v48 = a5;
          v53 = v23;
          v42 = ((__int64 (__fastcall *)(__int64, __int64, __int64))v41)(v30, v35, v40);
          v23 = v53;
          a5 = v48;
          v22 = v45;
          if ( !v42 )
            goto LABEL_34;
        }
LABEL_51:
        v29 = *(_QWORD *)(v29 + 8);
        if ( !v29 )
        {
          v27 = v44;
          goto LABEL_31;
        }
        v31 = *(unsigned __int8 **)(v29 + 24);
        v33 = *v31;
      }
      if ( v33 != 69 )
        goto LABEL_34;
      if ( v32 == *((_QWORD *)v31 + 1) )
        goto LABEL_51;
      v21 += 8LL;
      if ( v22 != (_BYTE *)v21 )
        goto LABEL_35;
LABEL_57:
      v59 = v23;
      v19 = v47;
      v5 = v58;
      if ( !v23 )
      {
LABEL_65:
        sub_2D57BD0((__int64 *)a2, v62);
        sub_9C95B0(a4, v19);
        if ( v66 != v68 )
          _libc_free((unsigned __int64)v66);
        goto LABEL_18;
      }
      if ( v66 != v68 )
        _libc_free((unsigned __int64)v66);
      if ( v69 != v71 )
        _libc_free((unsigned __int64)v69);
      v49 = v59;
LABEL_20:
      if ( (__int64 *)v55 == ++v64 )
        return v49;
    }
  }
  return 0;
}
