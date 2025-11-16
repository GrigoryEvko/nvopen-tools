// Function: sub_381DEB0
// Address: 0x381deb0
//
void __fastcall sub_381DEB0(__int64 *a1, __int64 a2, unsigned int *a3, __int64 a4, __m128i a5)
{
  unsigned int v5; // r15d
  __int64 v9; // rsi
  __int64 v10; // rax
  unsigned __int16 v11; // dx
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rdx
  char v19; // al
  unsigned int v20; // eax
  unsigned int v21; // eax
  __int64 v22; // r14
  int v23; // eax
  unsigned __int64 v24; // rdx
  _QWORD *v25; // r12
  __int128 v26; // rax
  __int64 v27; // r9
  unsigned int v28; // edx
  __int64 v29; // r12
  __int64 (__fastcall *v30)(__int64, __int64, unsigned int); // r14
  __int64 v31; // rax
  int v32; // edx
  unsigned __int16 v33; // ax
  __int128 v34; // rax
  __int64 v35; // r9
  int v36; // edx
  __int64 v37; // rax
  __int128 v38; // rax
  __int64 v39; // r9
  int v40; // edx
  __int64 v41; // [rsp+0h] [rbp-D0h]
  unsigned int v42; // [rsp+8h] [rbp-C8h]
  __int64 v43; // [rsp+40h] [rbp-90h] BYREF
  int v44; // [rsp+48h] [rbp-88h]
  unsigned int v45; // [rsp+50h] [rbp-80h] BYREF
  __int64 v46; // [rsp+58h] [rbp-78h]
  unsigned int v47; // [rsp+60h] [rbp-70h] BYREF
  unsigned __int64 v48; // [rsp+68h] [rbp-68h]
  __int64 v49; // [rsp+70h] [rbp-60h] BYREF
  char v50; // [rsp+78h] [rbp-58h]
  __int64 v51; // [rsp+80h] [rbp-50h] BYREF
  __int64 v52; // [rsp+88h] [rbp-48h]
  __int64 v53; // [rsp+90h] [rbp-40h]
  __int64 v54; // [rsp+98h] [rbp-38h]

  v9 = *(_QWORD *)(a2 + 80);
  v43 = v9;
  if ( v9 )
    sub_B96E90((__int64)&v43, v9, 1);
  v44 = *(_DWORD *)(a2 + 72);
  sub_375E510((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), (__int64)a3, a4);
  v10 = *(_QWORD *)(*(_QWORD *)a3 + 48LL) + 16LL * a3[2];
  v11 = *(_WORD *)v10;
  v46 = *(_QWORD *)(v10 + 8);
  v12 = *(_QWORD *)(a2 + 40);
  LOWORD(v45) = v11;
  v13 = *(_QWORD *)(v12 + 40);
  v14 = *(_QWORD *)(v13 + 104);
  LOWORD(v13) = *(_WORD *)(v13 + 96);
  v48 = v14;
  LOWORD(v47) = v13;
  if ( v11 )
  {
    if ( v11 == 1 || (unsigned __int16)(v11 - 504) <= 7u )
      goto LABEL_47;
    v16 = 16LL * (v11 - 1);
    v15 = *(_QWORD *)&byte_444C4A0[v16];
    LOBYTE(v16) = byte_444C4A0[v16 + 8];
  }
  else
  {
    v15 = sub_3007260((__int64)&v45);
    v53 = v15;
    v54 = v16;
  }
  v51 = v15;
  LOBYTE(v52) = v16;
  v42 = sub_CA1930(&v51);
  if ( !(_WORD)v47 )
  {
    v51 = sub_3007260((__int64)&v47);
    v52 = v17;
    v18 = v51;
    v19 = v52;
    goto LABEL_7;
  }
  if ( (_WORD)v47 == 1 || (unsigned __int16)(v47 - 504) <= 7u )
LABEL_47:
    BUG();
  v37 = 16LL * ((unsigned __int16)v47 - 1);
  v18 = *(_QWORD *)&byte_444C4A0[v37];
  v19 = byte_444C4A0[v37 + 8];
LABEL_7:
  v49 = v18;
  v50 = v19;
  v20 = sub_CA1930(&v49);
  if ( v42 < v20 )
  {
    v21 = v20 - v42;
    v22 = a1[1];
    switch ( v21 )
    {
      case 1u:
        LOWORD(v23) = 2;
        break;
      case 2u:
        LOWORD(v23) = 3;
        break;
      case 4u:
        LOWORD(v23) = 4;
        break;
      case 8u:
        LOWORD(v23) = 5;
        break;
      case 0x10u:
        LOWORD(v23) = 6;
        break;
      case 0x20u:
        LOWORD(v23) = 7;
        break;
      case 0x40u:
        LOWORD(v23) = 8;
        break;
      case 0x80u:
        LOWORD(v23) = 9;
        break;
      default:
        v23 = sub_3007020(*(_QWORD **)(v22 + 64), v21);
        HIWORD(v5) = HIWORD(v23);
LABEL_38:
        LOWORD(v5) = v23;
        *(_QWORD *)&v38 = sub_33F7D60((_QWORD *)v22, v5, v24);
        *(_QWORD *)a4 = sub_3406EB0((_QWORD *)v22, 3u, (__int64)&v43, v45, v46, v39, *(_OWORD *)a4, v38);
        *(_DWORD *)(a4 + 8) = v40;
        goto LABEL_27;
    }
    v24 = 0;
    goto LABEL_38;
  }
  v25 = (_QWORD *)a1[1];
  *(_QWORD *)&v26 = sub_33F7D60(v25, v47, v48);
  *(_QWORD *)a3 = sub_3406EB0(v25, 3u, (__int64)&v43, v45, v46, v27, *(_OWORD *)a3, v26);
  a3[2] = v28;
  v29 = a1[1];
  v41 = *a1;
  v30 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)*a1 + 32LL);
  v31 = sub_2E79000(*(__int64 **)(v29 + 40));
  if ( v30 == sub_2D42F30 )
  {
    v32 = sub_AE2980(v31, 0)[1];
    v33 = 2;
    if ( v32 != 1 )
    {
      v33 = 3;
      if ( v32 != 2 )
      {
        v33 = 4;
        if ( v32 != 4 )
        {
          v33 = 5;
          if ( v32 != 8 )
          {
            v33 = 6;
            if ( v32 != 16 )
            {
              v33 = 7;
              if ( v32 != 32 )
              {
                v33 = 8;
                if ( v32 != 64 )
                  v33 = 9 * (v32 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v33 = v30(v41, v31, 0);
  }
  *(_QWORD *)&v34 = sub_3400BD0(v29, v42 - 1, (__int64)&v43, v33, 0, 0, a5, 0);
  *(_QWORD *)a4 = sub_3406EB0((_QWORD *)v29, 0xBFu, (__int64)&v43, v45, v46, v35, *(_OWORD *)a3, v34);
  *(_DWORD *)(a4 + 8) = v36;
LABEL_27:
  if ( v43 )
    sub_B91220((__int64)&v43, v43);
}
