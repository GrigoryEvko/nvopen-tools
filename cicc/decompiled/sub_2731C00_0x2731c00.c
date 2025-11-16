// Function: sub_2731C00
// Address: 0x2731c00
//
void __fastcall sub_2731C00(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4)
{
  __int64 *v4; // rbx
  __int64 v5; // r9
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 *v9; // r15
  __int64 v10; // r8
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rax
  char **v15; // rsi
  unsigned __int64 v16; // r8
  __int64 v17; // rdx
  __int64 v18; // rcx
  _QWORD *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rsi
  unsigned __int64 v22; // rcx
  char *v23; // rsi
  __int64 v24; // rcx
  unsigned __int64 v25; // rax
  __int64 v26; // r8
  unsigned __int64 v27; // rdx
  _QWORD *v28; // rdi
  __int64 v29; // rbx
  unsigned __int64 *v30; // r12
  char *v31; // rbx
  __int64 v32; // [rsp+10h] [rbp-3C0h]
  unsigned __int64 *v34; // [rsp+20h] [rbp-3B0h]
  __int64 v35; // [rsp+28h] [rbp-3A8h]
  int v37; // [rsp+38h] [rbp-398h]
  __int64 v38; // [rsp+38h] [rbp-398h]
  _QWORD *v39; // [rsp+38h] [rbp-398h]
  char *v40; // [rsp+38h] [rbp-398h]
  __int64 *v41; // [rsp+48h] [rbp-388h] BYREF
  _BYTE *v42; // [rsp+50h] [rbp-380h] BYREF
  unsigned int v43; // [rsp+58h] [rbp-378h]
  _BYTE *v44; // [rsp+60h] [rbp-370h] BYREF
  __int64 v45; // [rsp+68h] [rbp-368h]
  _BYTE v46[128]; // [rsp+70h] [rbp-360h] BYREF
  __int64 v47; // [rsp+F0h] [rbp-2E0h]
  __int64 v48; // [rsp+F8h] [rbp-2D8h]
  _QWORD v49[2]; // [rsp+100h] [rbp-2D0h] BYREF
  __int64 v50; // [rsp+110h] [rbp-2C0h] BYREF
  __int64 v51; // [rsp+118h] [rbp-2B8h]
  _BYTE v52[688]; // [rsp+120h] [rbp-2B0h] BYREF

  v4 = a2;
  v41 = a2;
  if ( (unsigned int)sub_27309A0(a1, a2, a3, (__int64)&v41) <= 1 )
    return;
  v50 = (__int64)v52;
  v6 = v41[18];
  v7 = v41[19];
  v51 = 0x400000000LL;
  v49[1] = v7;
  v8 = *(_QWORD *)(v6 + 8);
  v49[0] = v6;
  v9 = (__int64 *)(v6 + 24);
  v32 = v8;
  if ( a3 != a2 )
  {
    while ( 1 )
    {
      v21 = v4[18];
      LODWORD(v45) = *(_DWORD *)(v21 + 32);
      if ( (unsigned int)v45 > 0x40 )
        sub_C43780((__int64)&v44, (const void **)(v21 + 24));
      else
        v44 = *(_BYTE **)(v21 + 24);
      sub_C46B40((__int64)&v44, v9);
      v22 = (unsigned __int64)v44;
      v43 = v45;
      v42 = v44;
      v37 = v45;
      if ( (unsigned int)v45 <= 0x40 )
        goto LABEL_24;
      v34 = (unsigned __int64 *)v44;
      if ( v37 - (unsigned int)sub_C444A0((__int64)&v42) <= 0x40 )
        break;
LABEL_5:
      v11 = sub_AD8D80(v32, (__int64)&v42);
LABEL_6:
      v12 = v4[19];
      if ( v12 )
        v12 = *(_QWORD *)(v12 + 8);
      v13 = *((unsigned int *)v4 + 2);
      v44 = v46;
      v45 = 0x800000000LL;
      if ( (_DWORD)v13 )
      {
        v35 = v12;
        v38 = v11;
        sub_272D8A0((__int64)&v44, (char **)v4, v11, v13, v10, v5);
        v12 = v35;
        v11 = v38;
      }
      v48 = v12;
      v14 = (unsigned int)v51;
      v15 = &v44;
      v47 = v11;
      v16 = (unsigned int)v51 + 1LL;
      v17 = v50;
      v18 = (unsigned int)v51;
      if ( v16 > HIDWORD(v51) )
      {
        if ( v50 > (unsigned __int64)&v44 || (unsigned __int64)&v44 >= v50 + 160 * (unsigned __int64)(unsigned int)v51 )
        {
          sub_2366E20((__int64)&v50, (unsigned int)v51 + 1LL, v50, (unsigned int)v51, v16, v5);
          v14 = (unsigned int)v51;
          v17 = v50;
          v15 = &v44;
          v18 = (unsigned int)v51;
        }
        else
        {
          v40 = (char *)&v44 - v50;
          sub_2366E20((__int64)&v50, (unsigned int)v51 + 1LL, v50, (unsigned int)v51, v16, v5);
          v17 = v50;
          v15 = (char **)&v40[v50];
          v14 = (unsigned int)v51;
          v18 = (unsigned int)v51;
        }
      }
      v19 = (_QWORD *)(v17 + 160 * v14);
      if ( v19 )
      {
        v19[1] = 0x800000000LL;
        *v19 = v19 + 2;
        v20 = *((unsigned int *)v15 + 2);
        if ( (_DWORD)v20 )
        {
          v39 = v19;
          sub_272D8A0((__int64)v19, v15, v20, v18, v16, v5);
          v19 = v39;
        }
        LODWORD(v18) = v51;
        v19[18] = v15[18];
        v19[19] = v15[19];
      }
      LODWORD(v51) = v18 + 1;
      if ( v44 != v46 )
        _libc_free((unsigned __int64)v44);
      if ( v43 > 0x40 && v42 )
        j_j___libc_free_0_0((unsigned __int64)v42);
      v4 += 21;
      if ( v4 == a3 )
        goto LABEL_28;
    }
    v22 = *v34;
LABEL_24:
    v11 = 0;
    if ( !v22 )
      goto LABEL_6;
    goto LABEL_5;
  }
LABEL_28:
  v23 = (char *)v49;
  v24 = *(unsigned int *)(a4 + 8);
  v25 = *(_QWORD *)a4;
  v26 = v24 + 1;
  v27 = v24;
  if ( v24 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
  {
    if ( v25 > (unsigned __int64)v49 || (v27 = v25 + 672 * v24, (unsigned __int64)v49 >= v27) )
    {
      sub_23672E0(a4, v24 + 1, v27, v24, v26, v5);
      v24 = *(unsigned int *)(a4 + 8);
      v25 = *(_QWORD *)a4;
      v23 = (char *)v49;
      v27 = v24;
    }
    else
    {
      v31 = (char *)v49 - v25;
      sub_23672E0(a4, v24 + 1, v27, v24, v26, v5);
      v25 = *(_QWORD *)a4;
      v24 = *(unsigned int *)(a4 + 8);
      v23 = &v31[*(_QWORD *)a4];
      v27 = v24;
    }
  }
  v28 = (_QWORD *)(v25 + 672 * v24);
  if ( v28 )
  {
    *v28 = *(_QWORD *)v23;
    v28[1] = *((_QWORD *)v23 + 1);
    v28[2] = v28 + 4;
    v28[3] = 0x400000000LL;
    if ( *((_DWORD *)v23 + 6) )
      sub_2731860((__int64)(v28 + 2), (__int64)(v23 + 16), v27, v24, v26, v5);
    LODWORD(v27) = *(_DWORD *)(a4 + 8);
  }
  v29 = v50;
  *(_DWORD *)(a4 + 8) = v27 + 1;
  v30 = (unsigned __int64 *)(v29 + 160LL * (unsigned int)v51);
  if ( (unsigned __int64 *)v29 != v30 )
  {
    do
    {
      v30 -= 20;
      if ( (unsigned __int64 *)*v30 != v30 + 2 )
        _libc_free(*v30);
    }
    while ( (unsigned __int64 *)v29 != v30 );
    v30 = (unsigned __int64 *)v50;
  }
  if ( v30 != (unsigned __int64 *)v52 )
    _libc_free((unsigned __int64)v30);
}
