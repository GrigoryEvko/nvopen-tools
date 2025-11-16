// Function: sub_375B6E0
// Address: 0x375b6e0
//
void __fastcall sub_375B6E0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __int64 a8,
        __int64 a9,
        __int64 a10)
{
  __int64 v13; // rbx
  __int64 v14; // rsi
  __int64 v15; // r15
  _QWORD *v16; // rdi
  __int64 v17; // rbx
  int v18; // edx
  __int64 v19; // rdx
  unsigned __int16 v20; // ax
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rdx
  char v24; // al
  unsigned __int64 v25; // r14
  unsigned int v26; // eax
  unsigned __int16 *v27; // rax
  __int64 v28; // rax
  unsigned __int16 v29; // ax
  __int64 v30; // rax
  char v31; // cl
  __int64 v32; // rax
  unsigned __int64 v33; // rax
  unsigned __int16 v34; // dx
  _QWORD *v35; // r14
  __int64 v36; // rcx
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rax
  __int128 v40; // rax
  __int64 v41; // r9
  int v42; // edx
  int v43; // r9d
  unsigned __int8 *v44; // rax
  __int64 v45; // rsi
  int v46; // edx
  unsigned __int64 v47; // r14
  unsigned __int64 v48; // r14
  __int64 v49; // rax
  __int128 v50; // rax
  __int64 v51; // r9
  int v52; // edx
  _QWORD *v53; // r12
  __int128 v54; // rax
  __int64 v55; // r9
  unsigned __int8 *v56; // rax
  int v57; // edx
  __int64 v58; // [rsp+10h] [rbp-110h]
  __int64 (__fastcall *v59)(__int64, __int64, __int64, __int64); // [rsp+18h] [rbp-108h]
  __int64 v60; // [rsp+20h] [rbp-100h]
  __int64 v61; // [rsp+28h] [rbp-F8h]
  unsigned __int16 v62; // [rsp+28h] [rbp-F8h]
  __int64 v63; // [rsp+28h] [rbp-F8h]
  __int64 v64; // [rsp+28h] [rbp-F8h]
  __int128 v65; // [rsp+40h] [rbp-E0h]
  __int64 v66; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v67; // [rsp+A8h] [rbp-78h]
  __int64 v68; // [rsp+B0h] [rbp-70h] BYREF
  int v69; // [rsp+B8h] [rbp-68h]
  __int64 v70; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v71; // [rsp+C8h] [rbp-58h]
  __int64 v72; // [rsp+D0h] [rbp-50h]
  __int64 v73; // [rsp+D8h] [rbp-48h]
  __int64 v74; // [rsp+E0h] [rbp-40h] BYREF
  __int64 v75; // [rsp+E8h] [rbp-38h]

  v66 = a4;
  *(_QWORD *)&v65 = a2;
  *((_QWORD *)&v65 + 1) = a3;
  v67 = a5;
  v13 = (unsigned int)a3;
  v14 = *(_QWORD *)(a2 + 80);
  v15 = a2;
  v68 = v14;
  if ( v14 )
    sub_B96E90((__int64)&v68, v14, 1);
  v16 = (_QWORD *)a1[1];
  v69 = *(_DWORD *)(v15 + 72);
  if ( !(_BYTE)qword_5050E88
    || (unsigned int)(*(_DWORD *)(*v16 + 544LL) - 42) > 1
    || (_WORD)v66 != (_WORD)a8
    || (unsigned __int16)(v66 - 6) > 1u )
  {
    v17 = 16 * v13;
    *(_QWORD *)a6 = sub_33FAF80((__int64)v16, 216, (__int64)&v68, (unsigned int)v66, v67, a6, a7);
    *(_DWORD *)(a6 + 8) = v18;
    v19 = v17 + *(_QWORD *)(v15 + 48);
    v20 = *(_WORD *)v19;
    v21 = *(_QWORD *)(v19 + 8);
    LOWORD(v70) = v20;
    v71 = v21;
    if ( v20 )
    {
      if ( v20 == 1 || (unsigned __int16)(v20 - 504) <= 7u )
        goto LABEL_34;
      v49 = 16LL * (v20 - 1);
      v23 = *(_QWORD *)&byte_444C4A0[v49];
      v24 = byte_444C4A0[v49 + 8];
    }
    else
    {
      v72 = sub_3007260((__int64)&v70);
      v73 = v22;
      v23 = v72;
      v24 = v73;
    }
    v74 = v23;
    v25 = 0;
    LOBYTE(v75) = v24;
    v26 = sub_CA1930(&v74) - 1;
    if ( v26 )
    {
      _BitScanReverse(&v26, v26);
      v25 = (int)(32 - (v26 ^ 0x1F));
    }
    v58 = *a1;
    v27 = (unsigned __int16 *)(v17 + *(_QWORD *)(v15 + 48));
    v59 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)*a1 + 56LL);
    v60 = *((_QWORD *)v27 + 1);
    v61 = *v27;
    v28 = sub_2E79000(*(__int64 **)(a1[1] + 40));
    v29 = v59(v58, v28, v61, v60);
    if ( v29 > 1u )
    {
      v62 = v29;
      if ( (unsigned __int16)(v29 - 504) > 7u )
      {
        v30 = 16LL * (v29 - 1);
        v31 = byte_444C4A0[v30 + 8];
        v32 = *(_QWORD *)&byte_444C4A0[v30];
        LOBYTE(v75) = v31;
        v74 = v32;
        v33 = sub_CA1930(&v74);
        v34 = v62;
        if ( v33 < v25 )
        {
          v34 = 2;
          v47 = (((v25 >> 1) | v25) >> 2) | (v25 >> 1) | v25;
          v48 = (v47 >> 4) | v47;
          if ( v48 )
          {
            v34 = 3;
            if ( (_DWORD)v48 != 1 )
            {
              v34 = 4;
              if ( (_DWORD)v48 != 3 )
              {
                v34 = 5;
                if ( (_DWORD)v48 != 7 )
                {
                  v34 = 6;
                  if ( (_DWORD)v48 != 15 )
                  {
                    v34 = 7;
                    if ( (_DWORD)v48 != 31 )
                      v34 = 8 * ((_DWORD)v48 == 63);
                  }
                }
              }
            }
          }
        }
        v35 = (_QWORD *)a1[1];
        v36 = v34;
        if ( !(_WORD)v66 )
        {
          v63 = v34;
          v37 = sub_3007260((__int64)&v66);
          v36 = v63;
          v74 = v37;
          v75 = v38;
          goto LABEL_15;
        }
        if ( (_WORD)v66 != 1 && (unsigned __int16)(v66 - 504) > 7u )
        {
          v38 = 16LL * ((unsigned __int16)v66 - 1);
          v37 = *(_QWORD *)&byte_444C4A0[v38];
          LOBYTE(v38) = byte_444C4A0[v38 + 8];
LABEL_15:
          v64 = v36;
          LOBYTE(v71) = v38;
          v70 = v37;
          v39 = sub_CA1930(&v70);
          *(_QWORD *)&v40 = sub_3400BD0((__int64)v35, v39, (__int64)&v68, v64, 0, 0, a7, 0);
          *(_QWORD *)a10 = sub_3406EB0(
                             v35,
                             0xC0u,
                             (__int64)&v68,
                             *(unsigned __int16 *)(*(_QWORD *)(v15 + 48) + v17),
                             *(_QWORD *)(*(_QWORD *)(v15 + 48) + v17 + 8),
                             v41,
                             v65,
                             v40);
          *(_DWORD *)(a10 + 8) = v42;
          v44 = sub_33FAF80(a1[1], 216, (__int64)&v68, (unsigned int)a8, a9, v43, a7);
          v45 = v68;
          *(_QWORD *)a10 = v44;
          *(_DWORD *)(a10 + 8) = v46;
          if ( !v45 )
            return;
          goto LABEL_16;
        }
      }
    }
LABEL_34:
    BUG();
  }
  *(_QWORD *)&v50 = sub_3400D50((__int64)v16, 0, (__int64)&v68, 0, a7);
  *(_QWORD *)a6 = sub_3406EB0(v16, 0x35u, (__int64)&v68, (unsigned int)v66, v67, v51, v65, v50);
  *(_DWORD *)(a6 + 8) = v52;
  v53 = (_QWORD *)a1[1];
  *(_QWORD *)&v54 = sub_3400D50((__int64)v53, 1, (__int64)&v68, 0, a7);
  v56 = sub_3406EB0(v53, 0x35u, (__int64)&v68, (unsigned int)a8, a9, v55, v65, v54);
  v45 = v68;
  *(_QWORD *)a10 = v56;
  *(_DWORD *)(a10 + 8) = v57;
  if ( v45 )
LABEL_16:
    sub_B91220((__int64)&v68, v45);
}
