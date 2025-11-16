// Function: sub_382F8A0
// Address: 0x382f8a0
//
void __fastcall sub_382F8A0(__int64 *a1, __int64 a2, __int64 a3, unsigned int *a4, __m128i a5)
{
  __int64 v8; // r11
  __int64 (__fastcall *v9)(__int64, __int64, unsigned int, __int64); // r13
  __int16 *v10; // rax
  unsigned __int16 v11; // di
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // rsi
  unsigned __int64 *v15; // rcx
  unsigned __int64 v16; // rsi
  __int64 v17; // r9
  unsigned __int64 v18; // r11
  __int64 v19; // r10
  __int64 v20; // rax
  __int64 v21; // rdi
  int v22; // edx
  __int64 v23; // rax
  __int64 v24; // rdx
  int v25; // eax
  __int64 v26; // r15
  int v27; // r14d
  __int64 (__fastcall *v28)(__int64, __int64, unsigned int); // r12
  __int64 v29; // rdi
  int v30; // edx
  unsigned __int16 v31; // ax
  __int128 v32; // rax
  __int64 v33; // r9
  unsigned int v34; // edx
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // r10
  __int64 v41; // rdx
  __int64 v42; // rdx
  char v43; // al
  unsigned int v44; // eax
  __int64 v45; // r12
  __int64 v46; // rax
  unsigned __int64 v47; // rdx
  __int64 v48; // rdx
  __int64 v49; // rsi
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rdi
  unsigned __int16 *v53; // rdx
  __int64 v54; // r9
  unsigned int v55; // edx
  __int128 v56; // [rsp-10h] [rbp-120h]
  __int64 v57; // [rsp+8h] [rbp-108h]
  __int64 v58; // [rsp+10h] [rbp-100h]
  unsigned __int64 v59; // [rsp+28h] [rbp-E8h]
  __int64 v60; // [rsp+30h] [rbp-E0h]
  __int64 v61; // [rsp+30h] [rbp-E0h]
  unsigned int v63; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v64; // [rsp+78h] [rbp-98h]
  __int64 v65; // [rsp+80h] [rbp-90h] BYREF
  int v66; // [rsp+88h] [rbp-88h]
  unsigned __int16 v67; // [rsp+90h] [rbp-80h] BYREF
  __int64 v68; // [rsp+98h] [rbp-78h]
  __int64 v69; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v70; // [rsp+A8h] [rbp-68h]
  unsigned __int64 v71; // [rsp+B0h] [rbp-60h]
  __int64 v72; // [rsp+B8h] [rbp-58h]
  unsigned __int64 v73; // [rsp+C0h] [rbp-50h] BYREF
  __int64 v74; // [rsp+C8h] [rbp-48h]
  __int64 v75; // [rsp+D0h] [rbp-40h]

  v8 = *a1;
  v9 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v10 = *(__int16 **)(a2 + 48);
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  v13 = a1[1];
  if ( v9 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v73, v8, *(_QWORD *)(v13 + 64), v11, v12);
    LOWORD(v63) = v74;
    v64 = v75;
  }
  else
  {
    v63 = v9(v8, *(_QWORD *)(v13 + 64), v11, v12);
    v64 = v48;
  }
  v14 = *(_QWORD *)(a2 + 80);
  v65 = v14;
  if ( v14 )
    sub_B96E90((__int64)&v65, v14, 1);
  v15 = *(unsigned __int64 **)(a2 + 40);
  v66 = *(_DWORD *)(a2 + 72);
  v16 = *v15;
  v17 = v15[1];
  v18 = *v15;
  v19 = 16LL * *((unsigned int *)v15 + 2);
  v20 = v19 + *(_QWORD *)(*v15 + 48);
  v21 = *(_QWORD *)(v20 + 8);
  v67 = *(_WORD *)v20;
  v68 = v21;
  if ( v67 != (_WORD)v63 || !(_WORD)v63 && v64 != v21 )
  {
    v58 = v17;
    v59 = v18;
    v61 = v19;
    LOWORD(v69) = v63;
    v70 = v64;
    v73 = sub_2D5B750((unsigned __int16 *)&v69);
    v74 = v35;
    v36 = sub_2D5B750(&v67);
    v72 = v37;
    v71 = v36;
    LODWORD(v17) = v58;
    if ( (_BYTE)v37 )
    {
      if ( !(_BYTE)v74 )
        goto LABEL_23;
    }
    if ( v71 > v73 )
    {
LABEL_23:
      v38 = sub_37AE0F0((__int64)a1, v16, v58);
      sub_375BC20(a1, v38, v39, a3, (__int64)a4, a5);
      v69 = sub_2D5B750((unsigned __int16 *)&v63);
      v40 = *(_QWORD *)(v59 + 48) + v61;
      v70 = v41;
      LOWORD(v41) = *(_WORD *)v40;
      v74 = *(_QWORD *)(v40 + 8);
      LOWORD(v73) = v41;
      v71 = sub_2D5B750((unsigned __int16 *)&v73);
      v72 = v42;
      v43 = v42;
      v73 = v71 - v69;
      if ( v69 )
        v43 = v70;
      LOBYTE(v74) = v43;
      v44 = sub_CA1930(&v73);
      v45 = a1[1];
      switch ( v44 )
      {
        case 1u:
          LOWORD(v46) = 2;
          break;
        case 2u:
          LOWORD(v46) = 3;
          break;
        case 4u:
          LOWORD(v46) = 4;
          break;
        case 8u:
          LOWORD(v46) = 5;
          break;
        case 0x10u:
          LOWORD(v46) = 6;
          break;
        case 0x20u:
          LOWORD(v46) = 7;
          break;
        case 0x40u:
          LOWORD(v46) = 8;
          break;
        case 0x80u:
          LOWORD(v46) = 9;
          break;
        default:
          v46 = sub_3007020(*(_QWORD **)(v45 + 64), v44);
          v57 = v46;
LABEL_40:
          v49 = v57;
          LOWORD(v49) = v46;
          v50 = sub_33F7D60((_QWORD *)v45, v49, v47);
          v52 = v51;
          v53 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)a4 + 48LL) + 16LL * a4[2]);
          *((_QWORD *)&v56 + 1) = v52;
          *(_QWORD *)&v56 = v50;
          *(_QWORD *)a4 = sub_3406EB0(
                            (_QWORD *)v45,
                            0xDEu,
                            (__int64)&v65,
                            *v53,
                            *((_QWORD *)v53 + 1),
                            v54,
                            *(_OWORD *)a4,
                            v56);
          a4[2] = v55;
          goto LABEL_17;
      }
      v47 = 0;
      goto LABEL_40;
    }
  }
  *(_QWORD *)a3 = sub_33FAF80(a1[1], 213, (__int64)&v65, v63, v64, v17, a5);
  *(_DWORD *)(a3 + 8) = v22;
  v23 = sub_2D5B750((unsigned __int16 *)&v63);
  v74 = v24;
  v73 = v23;
  v25 = sub_CA1930(&v73);
  v26 = a1[1];
  v27 = v25;
  v60 = *a1;
  v28 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)*a1 + 32LL);
  v29 = sub_2E79000(*(__int64 **)(v26 + 40));
  if ( v28 == sub_2D42F30 )
  {
    v30 = sub_AE2980(v29, 0)[1];
    v31 = 2;
    if ( v30 != 1 )
    {
      v31 = 3;
      if ( v30 != 2 )
      {
        v31 = 4;
        if ( v30 != 4 )
        {
          v31 = 5;
          if ( v30 != 8 )
          {
            v31 = 6;
            if ( v30 != 16 )
            {
              v31 = 7;
              if ( v30 != 32 )
              {
                v31 = 8;
                if ( v30 != 64 )
                  v31 = 9 * (v30 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v31 = v28(v60, v29, 0);
  }
  *(_QWORD *)&v32 = sub_3400BD0(v26, (unsigned int)(v27 - 1), (__int64)&v65, v31, 0, 0, a5, 0);
  *(_QWORD *)a4 = sub_3406EB0((_QWORD *)v26, 0xBFu, (__int64)&v65, v63, v64, v33, *(_OWORD *)a3, v32);
  a4[2] = v34;
LABEL_17:
  if ( v65 )
    sub_B91220((__int64)&v65, v65);
}
