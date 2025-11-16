// Function: sub_3285210
// Address: 0x3285210
//
__int64 __fastcall sub_3285210(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r12
  const __m128i *v10; // rdx
  __int64 v11; // r15
  __int64 v12; // rax
  __int16 v13; // cx
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // r10
  __int64 v18; // r8
  int v19; // eax
  int v20; // eax
  bool v21; // r15
  int v22; // r11d
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  bool v26; // al
  __int64 v27; // rsi
  bool v28; // al
  int v29; // r11d
  unsigned __int64 v30; // rdx
  unsigned int v31; // eax
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rax
  int v35; // ecx
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 *v38; // rax
  __int128 v39; // rax
  int v40; // r9d
  int v41; // r11d
  unsigned __int64 v42; // rdx
  unsigned int v43; // eax
  int v44; // r11d
  unsigned __int16 *v45; // rax
  __int64 v46; // r8
  int v47; // ecx
  unsigned __int16 *v48; // rax
  __int64 v49; // r12
  int v50; // ebx
  __int128 v51; // rax
  int v52; // r9d
  int v53; // [rsp-E8h] [rbp-E8h]
  int v54; // [rsp-E4h] [rbp-E4h]
  int v55; // [rsp-E4h] [rbp-E4h]
  char v56; // [rsp-E4h] [rbp-E4h]
  __int64 v57; // [rsp-E0h] [rbp-E0h]
  unsigned __int64 v58; // [rsp-D8h] [rbp-D8h]
  __int64 v59; // [rsp-D0h] [rbp-D0h]
  __int64 v60; // [rsp-D0h] [rbp-D0h]
  __int128 v61; // [rsp-C8h] [rbp-C8h]
  __int64 v63; // [rsp-B8h] [rbp-B8h]
  __int64 v64; // [rsp-B0h] [rbp-B0h]
  __int64 v65; // [rsp-B0h] [rbp-B0h]
  __int64 v66; // [rsp-B0h] [rbp-B0h]
  __int64 v67; // [rsp-B0h] [rbp-B0h]
  __int64 v68; // [rsp-B0h] [rbp-B0h]
  int v70; // [rsp-B0h] [rbp-B0h]
  unsigned int v71; // [rsp-A8h] [rbp-A8h] BYREF
  __int64 v72; // [rsp-A0h] [rbp-A0h]
  __int64 *v73; // [rsp-98h] [rbp-98h] BYREF
  unsigned int v74; // [rsp-90h] [rbp-90h]
  __int64 v75; // [rsp-88h] [rbp-88h] BYREF
  unsigned int v76; // [rsp-80h] [rbp-80h]
  __int64 v77[2]; // [rsp-78h] [rbp-78h] BYREF
  __int64 v78[2]; // [rsp-68h] [rbp-68h] BYREF
  unsigned __int64 v79; // [rsp-58h] [rbp-58h] BYREF
  unsigned int v80; // [rsp-50h] [rbp-50h]
  unsigned __int64 v81; // [rsp-48h] [rbp-48h] BYREF
  unsigned int v82; // [rsp-40h] [rbp-40h]

  if ( ((*(_DWORD *)(a2 + 24) - 190) & 0xFFFFFFFD) != 0 )
    return 0;
  v8 = a3;
  if ( *(_DWORD *)(a3 + 24) == 186
    && (unsigned __int8)sub_33E2390(
                          a1,
                          *(_QWORD *)(*(_QWORD *)(a3 + 40) + 40LL),
                          *(_QWORD *)(*(_QWORD *)(a3 + 40) + 48LL),
                          1) )
  {
    v37 = *(_QWORD *)(v8 + 40);
    *(_QWORD *)a5 = *(_QWORD *)(v37 + 40);
    *(_DWORD *)(a5 + 8) = *(_DWORD *)(v37 + 48);
    v38 = *(__int64 **)(v8 + 40);
    v8 = *v38;
    a4 = *((_DWORD *)v38 + 2);
  }
  v10 = *(const __m128i **)(a2 + 40);
  v11 = v10->m128i_u32[2];
  v12 = *(_QWORD *)(v10->m128i_i64[0] + 48) + 16 * v11;
  v64 = v10->m128i_i64[0];
  v13 = *(_WORD *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  v61 = (__int128)_mm_loadu_si128(v10);
  LOWORD(v71) = v13;
  v72 = v14;
  v15 = v10[3].m128i_i64[0];
  v16 = sub_33DFBC0(v10[2].m128i_i64[1], v15, 0, 0);
  v17 = v64;
  v18 = v16;
  v19 = *(_DWORD *)(a2 + 24);
  if ( !v18 )
  {
LABEL_40:
    if ( v19 != 192 )
      goto LABEL_6;
    v20 = *(_DWORD *)(v8 + 24);
    goto LABEL_10;
  }
  if ( v19 != 192 )
  {
LABEL_6:
    if ( v19 == 190 )
    {
      v20 = *(_DWORD *)(v8 + 24);
      if ( v20 == 60 || v20 == 192 )
      {
        v22 = 192;
        v21 = v20 == 60;
LABEL_12:
        if ( *(_DWORD *)(v17 + 24) == v20 )
        {
          v23 = *(_QWORD *)(v8 + 40);
          v24 = *(_QWORD *)(v17 + 40);
          if ( *(_QWORD *)v24 == *(_QWORD *)v23 && *(_DWORD *)(v24 + 8) == *(_DWORD *)(v23 + 8) )
          {
            v57 = 16LL * a4;
            v25 = *(_QWORD *)(v8 + 48) + v57;
            if ( *(_WORD *)v25 == (_WORD)v71 && ((_WORD)v71 || *(_QWORD *)(v25 + 8) == v72) )
            {
              v65 = v18;
              v54 = v22;
              v63 = sub_33DFBC0(*(_QWORD *)(v24 + 40), *(_QWORD *)(v24 + 48), 0, 0);
              v59 = sub_33DFBC0(
                      *(_QWORD *)(*(_QWORD *)(v8 + 40) + 40LL),
                      *(_QWORD *)(*(_QWORD *)(v8 + 40) + 48LL),
                      0,
                      0);
              if ( v65 )
              {
                v26 = sub_9867B0(*(_QWORD *)(v65 + 96) + 24LL);
                if ( v63 )
                {
                  if ( !v26 )
                  {
                    v27 = *(_QWORD *)(v63 + 96);
                    v28 = sub_9867B0(v27 + 24);
                    if ( v59 )
                    {
                      if ( !v28 && !sub_9867B0(*(_QWORD *)(v59 + 96) + 24LL) )
                      {
                        v58 = (unsigned int)sub_32844A0((unsigned __int16 *)&v71, v27);
                        v66 = *(_QWORD *)(v65 + 96) + 24LL;
                        if ( !sub_AAD8D0(v66, v58) )
                        {
                          sub_9865C0((__int64)&v81, v66);
                          v29 = v54;
                          if ( v82 > 0x40 )
                          {
                            sub_C43D10((__int64)&v81);
                            v29 = v54;
                          }
                          else
                          {
                            v30 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v82;
                            if ( !v82 )
                              v30 = 0;
                            v81 = v30 & ~v81;
                          }
                          v55 = v29;
                          sub_C46250((__int64)&v81);
                          sub_C46A40((__int64)&v81, v58);
                          v31 = v82;
                          v82 = 0;
                          v74 = v31;
                          v73 = (__int64 *)v81;
                          sub_969240((__int64 *)&v81);
                          sub_9865C0((__int64)&v75, *(_QWORD *)(v59 + 96) + 24LL);
                          sub_9865C0((__int64)v77, *(_QWORD *)(v63 + 96) + 24LL);
                          sub_3260590((__int64)&v75, (__int64)v77, 0);
                          if ( v21 )
                          {
                            LODWORD(v32) = (_DWORD)v73;
                            if ( v74 > 0x40 )
                              v32 = *v73;
                            sub_9866F0((__int64)v78, v76, v32);
                            v80 = 1;
                            v79 = 0;
                            v82 = 1;
                            v81 = 0;
                            sub_C4BFE0((__int64)&v75, (__int64)v78, &v79, &v81);
                            if ( !sub_D94970((__int64)&v81, 0) || !sub_AAD8B0((__int64)&v79, v77) )
                            {
                              sub_969240((__int64 *)&v81);
                              sub_969240((__int64 *)&v79);
                              sub_969240(v78);
                              v33 = 0;
LABEL_34:
                              v67 = v33;
                              sub_969240(v77);
                              sub_969240(&v75);
                              sub_969240((__int64 *)&v73);
                              return v67;
                            }
                            sub_969240((__int64 *)&v81);
                            sub_969240((__int64 *)&v79);
                            sub_969240(v78);
                            v44 = v55;
                          }
                          else
                          {
                            sub_C44AB0((__int64)&v79, (__int64)&v73, v76);
                            v41 = v55;
                            if ( v80 > 0x40 )
                            {
                              sub_C43D10((__int64)&v79);
                              v41 = v55;
                            }
                            else
                            {
                              v42 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v80;
                              if ( !v80 )
                                v42 = 0;
                              v79 = v42 & ~v79;
                            }
                            v53 = v41;
                            sub_C46250((__int64)&v79);
                            sub_C45EE0((__int64)&v79, &v75);
                            v43 = v80;
                            v80 = 0;
                            v82 = v43;
                            v81 = v79;
                            v56 = sub_AAD8B0((__int64)v77, &v81);
                            sub_969240((__int64 *)&v81);
                            sub_969240((__int64 *)&v79);
                            v44 = v53;
                            if ( !v56 )
                            {
                              v33 = 0;
                              goto LABEL_34;
                            }
                          }
                          v70 = v44;
                          v45 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 48LL)
                                                   + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 48LL));
                          v46 = *((_QWORD *)v45 + 1);
                          v47 = *v45;
                          v48 = (unsigned __int16 *)(*(_QWORD *)(v8 + 48) + v57);
                          v49 = *((_QWORD *)v48 + 1);
                          v50 = *v48;
                          *(_QWORD *)&v51 = sub_34007B0(a1, (unsigned int)&v73, a6, v47, v46, 0, 0);
                          v33 = sub_3406EB0(a1, v70, a6, v50, v49, v52, v61, v51);
                          goto LABEL_34;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        return 0;
      }
    }
    return 0;
  }
  v20 = *(_DWORD *)(v8 + 24);
  if ( v20 == 56 )
  {
    v34 = *(_QWORD *)(v8 + 40);
    if ( *(_QWORD *)v34 != *(_QWORD *)(v34 + 40) )
      return 0;
    v35 = *(_DWORD *)(v34 + 8);
    LOBYTE(v15) = *(_DWORD *)(v34 + 48) == v35;
    if ( ((v64 == *(_QWORD *)v34) & (unsigned __int8)v15) == 0 || (_DWORD)v11 != v35 )
      return 0;
    v68 = v18;
    v60 = v17;
    v36 = sub_32844A0((unsigned __int16 *)&v71, v15);
    if ( sub_D94970(*(_QWORD *)(v68 + 96) + 24LL, (_QWORD *)(v36 - 1)) )
    {
      *(_QWORD *)&v39 = sub_3400E40(a1, 1, v71, v72, a6);
      return sub_3406EB0(a1, 190, a6, v71, v72, v40, v61, v39);
    }
    v19 = *(_DWORD *)(a2 + 24);
    v17 = v60;
    v18 = v68;
    goto LABEL_40;
  }
LABEL_10:
  v21 = v20 == 58;
  if ( v20 == 58 || v20 == 190 )
  {
    v22 = 190;
    goto LABEL_12;
  }
  return 0;
}
