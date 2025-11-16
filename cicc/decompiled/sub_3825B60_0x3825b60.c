// Function: sub_3825B60
// Address: 0x3825b60
//
void __fastcall sub_3825B60(__int64 *a1, __int64 a2, __int64 a3, unsigned int *a4, __m128i a5)
{
  __int64 v8; // rsi
  __int64 v9; // r9
  __int64 v10; // rsi
  __int64 v11; // rax
  unsigned __int16 v12; // dx
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int16 v16; // si
  __int64 v17; // r8
  int v18; // edx
  __int64 v19; // r15
  __int64 (__fastcall *v20)(__int64, __int64, unsigned int); // r13
  __int64 v21; // rax
  int v22; // edx
  unsigned __int16 v23; // ax
  unsigned int v24; // r13d
  __int64 v25; // rax
  __int16 v26; // dx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  unsigned __int8 *v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rdi
  unsigned __int16 *v33; // rdx
  __int64 v34; // r9
  unsigned int v35; // edx
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // rsi
  char v42; // al
  unsigned int v43; // eax
  __int64 v44; // r12
  int v45; // eax
  unsigned __int64 v46; // rdx
  unsigned int v47; // ecx
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rdi
  unsigned __int16 *v51; // rdx
  __int64 v52; // r9
  unsigned int v53; // edx
  __int128 v54; // [rsp-20h] [rbp-100h]
  __int128 v55; // [rsp-10h] [rbp-F0h]
  __int16 v56; // [rsp+Ah] [rbp-D6h]
  __int64 v57; // [rsp+10h] [rbp-D0h]
  __int64 v58; // [rsp+20h] [rbp-C0h]
  __int64 v59; // [rsp+28h] [rbp-B8h]
  __int64 v60; // [rsp+28h] [rbp-B8h]
  __int64 v61; // [rsp+60h] [rbp-80h] BYREF
  int v62; // [rsp+68h] [rbp-78h]
  unsigned __int16 v63; // [rsp+70h] [rbp-70h] BYREF
  __int64 v64; // [rsp+78h] [rbp-68h]
  __int64 v65; // [rsp+80h] [rbp-60h] BYREF
  __int64 v66; // [rsp+88h] [rbp-58h]
  unsigned __int64 v67; // [rsp+90h] [rbp-50h]
  __int64 v68; // [rsp+98h] [rbp-48h]
  unsigned __int64 v69; // [rsp+A0h] [rbp-40h] BYREF
  __int64 v70; // [rsp+A8h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 80);
  v61 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v61, v8, 1);
  v62 = *(_DWORD *)(a2 + 72);
  sub_375E510((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), a3, (__int64)a4);
  v9 = *(_QWORD *)(a2 + 40);
  v10 = *(_QWORD *)a3;
  v11 = *(_QWORD *)(v9 + 40);
  v12 = *(_WORD *)(v11 + 96);
  v13 = *(_QWORD *)(v11 + 104);
  v14 = *(unsigned int *)(a3 + 8);
  v64 = v13;
  v15 = *(_QWORD *)(v10 + 48) + 16 * v14;
  v63 = v12;
  v16 = *(_WORD *)v15;
  v17 = *(_QWORD *)(v15 + 8);
  if ( v12 != *(_WORD *)v15 || !v16 && v17 != v13 )
  {
    v57 = v9;
    LOWORD(v65) = *(_WORD *)v15;
    v66 = v17;
    v58 = v17;
    v69 = sub_2D5B750((unsigned __int16 *)&v65);
    v70 = v36;
    v37 = sub_2D5B750(&v63);
    v17 = v58;
    v68 = v38;
    v67 = v37;
    v9 = v57;
    if ( (_BYTE)v38 )
    {
      if ( !(_BYTE)v70 )
        goto LABEL_21;
    }
    if ( v67 > v69 )
    {
LABEL_21:
      LOWORD(v69) = v16;
      v70 = v58;
      v60 = sub_2D5B750((unsigned __int16 *)&v69);
      v68 = v39;
      v67 = v60;
      v41 = sub_2D5B750(&v63);
      v66 = v40;
      v42 = v68;
      v65 = v41;
      v69 = v41 - v60;
      if ( !v60 )
        v42 = v40;
      LOBYTE(v70) = v42;
      v43 = sub_CA1930(&v69);
      v44 = a1[1];
      switch ( v43 )
      {
        case 1u:
          LOWORD(v45) = 2;
          break;
        case 2u:
          LOWORD(v45) = 3;
          break;
        case 4u:
          LOWORD(v45) = 4;
          break;
        case 8u:
          LOWORD(v45) = 5;
          break;
        case 0x10u:
          LOWORD(v45) = 6;
          break;
        case 0x20u:
          LOWORD(v45) = 7;
          break;
        case 0x40u:
          LOWORD(v45) = 8;
          break;
        case 0x80u:
          LOWORD(v45) = 9;
          break;
        default:
          v45 = sub_3007020(*(_QWORD **)(v44 + 64), v43);
          v56 = HIWORD(v45);
LABEL_37:
          HIWORD(v47) = v56;
          LOWORD(v47) = v45;
          v48 = sub_33F7D60((_QWORD *)v44, v47, v46);
          v50 = v49;
          v51 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)a4 + 48LL) + 16LL * a4[2]);
          *((_QWORD *)&v55 + 1) = v50;
          *(_QWORD *)&v55 = v48;
          *(_QWORD *)a4 = sub_3406EB0(
                            (_QWORD *)v44,
                            0xDEu,
                            (__int64)&v61,
                            *v51,
                            *((_QWORD *)v51 + 1),
                            v52,
                            *(_OWORD *)a4,
                            v55);
          a4[2] = v53;
          goto LABEL_15;
      }
      v46 = 0;
      goto LABEL_37;
    }
  }
  *(_QWORD *)a3 = sub_3406EB0((_QWORD *)a1[1], 0xDEu, (__int64)&v61, v16, v17, v9, *(_OWORD *)a3, *(_OWORD *)(v9 + 40));
  *(_DWORD *)(a3 + 8) = v18;
  v19 = a1[1];
  v59 = *a1;
  v20 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)*a1 + 32LL);
  v21 = sub_2E79000(*(__int64 **)(v19 + 40));
  if ( v20 == sub_2D42F30 )
  {
    v22 = sub_AE2980(v21, 0)[1];
    v23 = 2;
    if ( v22 != 1 )
    {
      v23 = 3;
      if ( v22 != 2 )
      {
        v23 = 4;
        if ( v22 != 4 )
        {
          v23 = 5;
          if ( v22 != 8 )
          {
            v23 = 6;
            if ( v22 != 16 )
            {
              v23 = 7;
              if ( v22 != 32 )
              {
                v23 = 8;
                if ( v22 != 64 )
                  v23 = 9 * (v22 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v23 = v20(v59, v21, 0);
  }
  v24 = v23;
  v25 = *(_QWORD *)(*(_QWORD *)a4 + 48LL) + 16LL * a4[2];
  v26 = *(_WORD *)v25;
  v27 = *(_QWORD *)(v25 + 8);
  LOWORD(v69) = v26;
  v70 = v27;
  v65 = sub_2D5B750((unsigned __int16 *)&v69);
  v66 = v28;
  v69 = v65;
  LOBYTE(v70) = v28;
  v29 = sub_CA1930(&v69);
  v30 = sub_3400BD0(v19, v29 - 1, (__int64)&v61, v24, 0, 0, a5, 0);
  v32 = v31;
  v33 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)a4 + 48LL) + 16LL * a4[2]);
  *((_QWORD *)&v54 + 1) = v32;
  *(_QWORD *)&v54 = v30;
  *(_QWORD *)a4 = sub_3406EB0((_QWORD *)v19, 0xBFu, (__int64)&v61, *v33, *((_QWORD *)v33 + 1), v34, *(_OWORD *)a3, v54);
  a4[2] = v35;
LABEL_15:
  if ( v61 )
    sub_B91220((__int64)&v61, v61);
}
