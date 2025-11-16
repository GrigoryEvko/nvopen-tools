// Function: sub_26FAF90
// Address: 0x26faf90
//
__int64 __fastcall sub_26FAF90(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  _QWORD *v6; // rbx
  __int64 v8; // rdx
  __int64 v10; // rcx
  __int64 v11; // rsi
  _QWORD *v12; // rax
  __int64 result; // rax
  __int64 v14; // r12
  __int64 v15; // rax
  unsigned __int8 *v16; // r12
  _BYTE *v17; // r8
  __int64 (__fastcall *v18)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v19; // rax
  __int64 v20; // r9
  _QWORD *v21; // r13
  __int64 **v22; // r12
  __int64 (__fastcall *v23)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v24; // r15
  _QWORD *v25; // r12
  __int64 v26; // r13
  _QWORD *v27; // rdi
  _DWORD *v28; // rax
  _QWORD **v29; // rdx
  int v30; // ecx
  __int64 *v31; // rax
  __int64 v32; // rax
  __int64 v33; // r15
  _BYTE *v34; // r12
  __int64 v35; // rdx
  unsigned int v36; // esi
  _QWORD *v37; // rax
  __int64 v38; // r13
  _BYTE *v39; // r12
  __int64 v40; // rdx
  unsigned int v41; // esi
  __int64 v42; // rax
  _BYTE *v43; // [rsp+0h] [rbp-1C0h]
  _BYTE *v46; // [rsp+18h] [rbp-1A8h]
  __int64 v47; // [rsp+18h] [rbp-1A8h]
  _BYTE *v48; // [rsp+18h] [rbp-1A8h]
  unsigned int v49; // [rsp+24h] [rbp-19Ch]
  __int64 v50; // [rsp+30h] [rbp-190h]
  _QWORD *v52; // [rsp+58h] [rbp-168h]
  __int64 v53; // [rsp+68h] [rbp-158h]
  _BYTE v54[32]; // [rsp+70h] [rbp-150h] BYREF
  __int16 v55; // [rsp+90h] [rbp-130h]
  _BYTE v56[32]; // [rsp+A0h] [rbp-120h] BYREF
  __int16 v57; // [rsp+C0h] [rbp-100h]
  int v58[8]; // [rsp+D0h] [rbp-F0h] BYREF
  __int16 v59; // [rsp+F0h] [rbp-D0h]
  _BYTE *v60; // [rsp+100h] [rbp-C0h] BYREF
  __int64 v61; // [rsp+108h] [rbp-B8h]
  _BYTE v62[32]; // [rsp+110h] [rbp-B0h] BYREF
  __int64 v63; // [rsp+130h] [rbp-90h]
  __int64 v64; // [rsp+138h] [rbp-88h]
  __int64 v65; // [rsp+140h] [rbp-80h]
  __int64 v66; // [rsp+148h] [rbp-78h]
  void **v67; // [rsp+150h] [rbp-70h]
  void **v68; // [rsp+158h] [rbp-68h]
  __int64 v69; // [rsp+160h] [rbp-60h]
  int v70; // [rsp+168h] [rbp-58h]
  __int16 v71; // [rsp+16Ch] [rbp-54h]
  char v72; // [rsp+16Eh] [rbp-52h]
  __int64 v73; // [rsp+170h] [rbp-50h]
  __int64 v74; // [rsp+178h] [rbp-48h]
  void *v75; // [rsp+180h] [rbp-40h] BYREF
  void *v76; // [rsp+188h] [rbp-38h] BYREF

  v6 = *(_QWORD **)a2;
  v52 = *(_QWORD **)(a2 + 8);
  if ( *(_QWORD **)a2 != v52 )
  {
    v50 = a1 + 176;
    v8 = (unsigned int)((_BYTE)a5 == 0) + 32;
    v10 = (__int64)v62;
    v49 = ((_BYTE)a5 == 0) + 32;
    v43 = a3;
    while ( 1 )
    {
      v11 = v6[1];
      if ( !*(_BYTE *)(a1 + 204) )
        goto LABEL_12;
      v12 = *(_QWORD **)(a1 + 184);
      v10 = *(unsigned int *)(a1 + 196);
      v8 = (__int64)&v12[v10];
      if ( v12 != (_QWORD *)v8 )
      {
        while ( v11 != *v12 )
        {
          if ( (_QWORD *)v8 == ++v12 )
            goto LABEL_35;
        }
        goto LABEL_8;
      }
LABEL_35:
      if ( (unsigned int)v10 < *(_DWORD *)(a1 + 192) )
      {
        *(_DWORD *)(a1 + 196) = v10 + 1;
        *(_QWORD *)v8 = v11;
        ++*(_QWORD *)(a1 + 176);
LABEL_13:
        v14 = v6[1];
        v15 = sub_BD5C60(v14);
        v72 = 7;
        v66 = v15;
        v67 = &v75;
        v68 = &v76;
        v60 = v62;
        v61 = 0x200000000LL;
        v75 = &unk_49DA100;
        v71 = 512;
        LOWORD(v65) = 0;
        v76 = &unk_49DA0B0;
        v69 = 0;
        v70 = 0;
        v73 = 0;
        v74 = 0;
        v63 = 0;
        v64 = 0;
        sub_D5F1F0((__int64)&v60, v14);
        v57 = 257;
        v55 = 257;
        v16 = (unsigned __int8 *)sub_26FAB50(
                                   (__int64 *)&v60,
                                   0x31u,
                                   a6,
                                   *(__int64 ***)(*v6 + 8LL),
                                   (__int64)v54,
                                   0,
                                   v58[0],
                                   0);
        v17 = (_BYTE *)*v6;
        v18 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, unsigned __int8 *))*((_QWORD *)*v67 + 7);
        if ( v18 != sub_928890 )
        {
          v48 = (_BYTE *)*v6;
          v42 = v18((__int64)v67, v49, v17, v16);
          v17 = v48;
          v21 = (_QWORD *)v42;
LABEL_17:
          if ( v21 )
            goto LABEL_18;
          goto LABEL_37;
        }
        if ( *v17 <= 0x15u && *v16 <= 0x15u )
        {
          v46 = (_BYTE *)*v6;
          v19 = sub_AAB310(v49, (unsigned __int8 *)*v6, v16);
          v17 = v46;
          v21 = (_QWORD *)v19;
          goto LABEL_17;
        }
LABEL_37:
        v47 = (__int64)v17;
        v59 = 257;
        v21 = sub_BD2C40(72, unk_3F10FD0);
        if ( v21 )
        {
          v29 = *(_QWORD ***)(v47 + 8);
          v30 = *((unsigned __int8 *)v29 + 8);
          if ( (unsigned int)(v30 - 17) > 1 )
          {
            v32 = sub_BCB2A0(*v29);
          }
          else
          {
            BYTE4(v53) = (_BYTE)v30 == 18;
            LODWORD(v53) = *((_DWORD *)v29 + 8);
            v31 = (__int64 *)sub_BCB2A0(*v29);
            v32 = sub_BCE1B0(v31, v53);
          }
          sub_B523C0((__int64)v21, v32, 53, v49, v47, (__int64)v16, (__int64)v58, 0, 0, 0);
        }
        (*((void (__fastcall **)(void **, _QWORD *, _BYTE *, __int64, __int64))*v68 + 2))(v68, v21, v56, v64, v65);
        v33 = (__int64)v60;
        v34 = &v60[16 * (unsigned int)v61];
        if ( v60 != v34 )
        {
          do
          {
            v35 = *(_QWORD *)(v33 + 8);
            v36 = *(_DWORD *)v33;
            v33 += 16;
            sub_B99FD0((__int64)v21, v36, v35);
          }
          while ( v34 != (_BYTE *)v33 );
        }
LABEL_18:
        v57 = 257;
        v22 = *(__int64 ***)(v6[1] + 8LL);
        if ( v22 == (__int64 **)v21[1] )
        {
          v24 = (__int64)v21;
          goto LABEL_24;
        }
        v23 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, __int64))*((_QWORD *)*v67 + 15);
        if ( v23 != sub_920130 )
        {
          v24 = ((__int64 (__fastcall *)(void **, __int64, _QWORD *, __int64 **, _BYTE *))v23)(v67, 39, v21, v22, v17);
          goto LABEL_23;
        }
        if ( *(_BYTE *)v21 <= 0x15u )
        {
          if ( (unsigned __int8)sub_AC4810(0x27u) )
            v24 = sub_ADAB70(39, (unsigned __int64)v21, v22, 0);
          else
            v24 = sub_AA93C0(0x27u, (unsigned __int64)v21, (__int64)v22);
LABEL_23:
          if ( v24 )
            goto LABEL_24;
        }
        v59 = 257;
        v37 = sub_BD2C40(72, unk_3F10A14);
        v24 = (__int64)v37;
        if ( v37 )
          sub_B515B0((__int64)v37, (__int64)v21, (__int64)v22, (__int64)v58, 0, 0);
        (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v68 + 2))(v68, v24, v56, v64, v65);
        v38 = (__int64)v60;
        v39 = &v60[16 * (unsigned int)v61];
        if ( v60 != v39 )
        {
          do
          {
            v40 = *(_QWORD *)(v38 + 8);
            v41 = *(_DWORD *)v38;
            v38 += 16;
            sub_B99FD0(v24, v41, v40);
          }
          while ( v39 != (_BYTE *)v38 );
        }
LABEL_24:
        if ( *(_BYTE *)(a1 + 104) )
          sub_26F96D0(
            (__int64)v6,
            "unique-ret-val",
            14,
            v43,
            a4,
            v20,
            *(__int64 (__fastcall **)(__int64, __int64))(a1 + 112),
            *(_QWORD *)(a1 + 120));
        sub_BD84D0(v6[1], v24);
        v25 = (_QWORD *)v6[1];
        if ( *(_BYTE *)v25 == 34 )
        {
          v26 = *(v25 - 12);
          v27 = sub_BD2C40(72, 1u);
          if ( v27 )
            sub_B4C8F0((__int64)v27, v26, 1u, (__int64)(v25 + 3), 0);
          sub_AA5980(*(v25 - 8), v25[5], 0);
          v25 = (_QWORD *)v6[1];
        }
        sub_B43D60(v25);
        v28 = (_DWORD *)v6[2];
        if ( v28 )
          --*v28;
        nullsub_61();
        v75 = &unk_49DA100;
        nullsub_63();
        if ( v60 == v62 )
          goto LABEL_8;
        _libc_free((unsigned __int64)v60);
        v6 += 3;
        if ( v52 == v6 )
          break;
      }
      else
      {
LABEL_12:
        sub_C8CC70(v50, v11, v8, v10, a5, a6);
        if ( (_BYTE)v8 )
          goto LABEL_13;
LABEL_8:
        v6 += 3;
        if ( v52 == v6 )
          break;
      }
    }
  }
  *(_BYTE *)(a2 + 24) = 1;
  result = *(_QWORD *)(a2 + 32);
  if ( result != *(_QWORD *)(a2 + 40) )
    *(_QWORD *)(a2 + 40) = result;
  return result;
}
