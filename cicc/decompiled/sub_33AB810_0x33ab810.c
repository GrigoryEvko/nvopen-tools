// Function: sub_33AB810
// Address: 0x33ab810
//
_QWORD *__fastcall sub_33AB810(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v5; // r13
  __int64 v6; // rdx
  __int64 v7; // r9
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rsi
  _BYTE *v11; // rax
  int v12; // r8d
  _BYTE *v13; // rdx
  _BYTE *v14; // r14
  _BYTE *i; // rdx
  __int64 v16; // rbx
  __int64 v17; // r14
  __int64 (__fastcall *v18)(__int64, __int64, unsigned int); // rax
  __int16 v19; // r13
  int v20; // eax
  int v21; // eax
  __int64 v22; // rax
  int v23; // edx
  int v24; // edi
  __int64 v25; // rdx
  __int64 v26; // rax
  char v27; // r15
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r13
  __int128 v31; // rax
  __int64 v32; // rsi
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  int v37; // eax
  int v38; // edx
  int v39; // ecx
  int v40; // r8d
  __int64 v41; // rax
  unsigned int v42; // edx
  unsigned int v43; // r13d
  __int64 v44; // r14
  __int64 v45; // r8
  __int64 v46; // r15
  _QWORD *result; // rax
  __int64 v48; // r14
  __int64 v49; // r13
  __int128 v50; // rax
  __int64 v51; // rsi
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  int v56; // edx
  int v57; // r8d
  int v58; // r9d
  __int64 v59; // rax
  __int64 v60; // rsi
  __int64 v61; // r15
  __int64 v62; // rax
  int v63; // eax
  int v64; // edx
  __int64 v65; // r15
  unsigned int v66; // edx
  __int64 v67; // r8
  __int64 v68; // rax
  unsigned int v69; // eax
  __int64 v70; // rdx
  __int64 v71; // r10
  unsigned int v72; // r12d
  __int64 v73; // rax
  __int64 v74; // r9
  bool v75; // zf
  __int64 v76; // rsi
  unsigned int v77; // edx
  int v78; // [rsp+8h] [rbp-228h]
  int v79; // [rsp+10h] [rbp-220h]
  __int64 v80; // [rsp+38h] [rbp-1F8h]
  __int128 v81; // [rsp+40h] [rbp-1F0h]
  __int128 v82; // [rsp+50h] [rbp-1E0h]
  int v83; // [rsp+50h] [rbp-1E0h]
  int v84; // [rsp+50h] [rbp-1E0h]
  int v85; // [rsp+58h] [rbp-1D8h]
  int v86; // [rsp+58h] [rbp-1D8h]
  __int64 v88; // [rsp+60h] [rbp-1D0h]
  int v89; // [rsp+60h] [rbp-1D0h]
  __int64 v90; // [rsp+60h] [rbp-1D0h]
  __int128 v91; // [rsp+60h] [rbp-1D0h]
  __int64 v92; // [rsp+60h] [rbp-1D0h]
  __int64 v93; // [rsp+70h] [rbp-1C0h]
  __int64 v94; // [rsp+70h] [rbp-1C0h]
  __int64 v95; // [rsp+70h] [rbp-1C0h]
  __int64 v96; // [rsp+70h] [rbp-1C0h]
  __int128 v97; // [rsp+70h] [rbp-1C0h]
  __int64 v98; // [rsp+70h] [rbp-1C0h]
  __int64 v99; // [rsp+70h] [rbp-1C0h]
  int v100; // [rsp+70h] [rbp-1C0h]
  __int64 v102; // [rsp+E0h] [rbp-150h] BYREF
  int v103; // [rsp+E8h] [rbp-148h]
  unsigned __int64 v104; // [rsp+F0h] [rbp-140h] BYREF
  char v105; // [rsp+F8h] [rbp-138h]
  __int64 v106; // [rsp+100h] [rbp-130h] BYREF
  __int64 v107; // [rsp+108h] [rbp-128h]
  _BYTE *v108; // [rsp+110h] [rbp-120h] BYREF
  __int64 v109; // [rsp+118h] [rbp-118h]
  _BYTE v110[64]; // [rsp+120h] [rbp-110h] BYREF
  unsigned __int64 v111[2]; // [rsp+160h] [rbp-D0h] BYREF
  _BYTE v112[64]; // [rsp+170h] [rbp-C0h] BYREF
  _BYTE *v113; // [rsp+1B0h] [rbp-80h] BYREF
  __int64 v114; // [rsp+1B8h] [rbp-78h]
  _BYTE v115[112]; // [rsp+1C0h] [rbp-70h] BYREF

  v2 = a1;
  v3 = *(_QWORD *)(a1 + 864);
  v4 = *(_QWORD *)(v3 + 16);
  v5 = sub_2E79000(*(__int64 **)(v3 + 40));
  if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 856) + 544LL) - 42) > 1 )
  {
    v48 = *(_QWORD *)(a1 + 864);
    v49 = 1LL << sub_AE5020(v5, *(_QWORD *)(a2 + 8));
    *(_QWORD *)&v50 = sub_33F1270(*(_QWORD *)(a1 + 864), *(_QWORD *)(a2 - 32));
    v51 = *(_QWORD *)(a2 - 32);
    v91 = v50;
    *(_QWORD *)&v97 = sub_338B750(a1, v51);
    *((_QWORD *)&v97 + 1) = v52;
    v113 = 0;
    v57 = sub_33738B0(a1, v51, v52, v53, v54, v55);
    v58 = v56;
    v59 = *(_QWORD *)a1;
    LODWORD(v114) = *(_DWORD *)(a1 + 848);
    if ( v59 )
    {
      if ( &v113 != (_BYTE **)(v59 + 48) )
      {
        v60 = *(_QWORD *)(v59 + 48);
        v113 = (_BYTE *)v60;
        if ( v60 )
        {
          v83 = v57;
          v85 = v56;
          sub_B96E90((__int64)&v113, v60, 1);
          v57 = v83;
          v58 = v85;
        }
      }
    }
    v84 = v57;
    v86 = v58;
    v61 = *(_QWORD *)(a2 + 8);
    v62 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL));
    v63 = sub_336EEB0(v4, v62, v61, 0);
    v65 = sub_3411830(v48, v63, v64, (unsigned int)&v113, v84, v86, v97, v91, v49);
    v43 = v66;
    v44 = v65;
    if ( v113 )
      sub_B91220((__int64)&v113, (__int64)v113);
    v67 = *(_QWORD *)(a1 + 864);
    if ( v65 )
    {
      v98 = *(_QWORD *)(a1 + 864);
      nullsub_1875(v65, v98, 0);
      *(_QWORD *)(v98 + 384) = v65;
      *(_DWORD *)(v98 + 392) = 1;
      sub_33E2B60(v98, 0);
    }
    else
    {
      *(_QWORD *)(v67 + 384) = 0;
      *(_DWORD *)(v67 + 392) = 1;
    }
  }
  else
  {
    LOBYTE(v114) = 0;
    v108 = v110;
    v109 = 0x400000000LL;
    v111[1] = 0x400000000LL;
    v111[0] = (unsigned __int64)v112;
    v6 = *(_QWORD *)(a2 + 8);
    v113 = 0;
    sub_34B8C80(v4, v5, v6, (unsigned int)&v108, 0, (unsigned int)v111, __PAIR128__(v114, 0));
    v8 = *(_DWORD *)(a1 + 848);
    v9 = *(_QWORD *)a1;
    v102 = 0;
    v103 = v8;
    if ( v9 )
    {
      if ( &v102 != (__int64 *)(v9 + 48) )
      {
        v10 = *(_QWORD *)(v9 + 48);
        v102 = v10;
        if ( v10 )
          sub_B96E90((__int64)&v102, v10, 1);
      }
    }
    v11 = v115;
    v114 = 0x400000000LL;
    v12 = v109;
    v13 = v115;
    v14 = v115;
    v80 = (unsigned int)v109;
    v113 = v115;
    if ( (_DWORD)v109 )
    {
      if ( (unsigned int)v109 > 4uLL )
      {
        v100 = v109;
        sub_C8D5F0((__int64)&v113, v115, (unsigned int)v109, 0x10u, (unsigned int)v109, v7);
        v13 = v113;
        v12 = v100;
        v11 = &v113[16 * (unsigned int)v114];
      }
      for ( i = &v13[16 * v80]; i != v11; v11 += 16 )
      {
        if ( v11 )
        {
          *(_QWORD *)v11 = 0;
          *((_DWORD *)v11 + 2) = 0;
        }
      }
      LODWORD(v114) = v12;
      v93 = v5;
      v16 = 0;
      do
      {
        v17 = *(_QWORD *)(a1 + 864);
        v18 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v4 + 32LL);
        if ( v18 == sub_2D42F30 )
        {
          v19 = 2;
          v20 = sub_AE2980(v93, 0)[1];
          if ( v20 != 1 )
          {
            v19 = 3;
            if ( v20 != 2 )
            {
              v19 = 4;
              if ( v20 != 4 )
              {
                v19 = 5;
                if ( v20 != 8 )
                {
                  v19 = 6;
                  if ( v20 != 16 )
                  {
                    v19 = 7;
                    if ( v20 != 32 )
                    {
                      v19 = 8;
                      if ( v20 != 64 )
                        v19 = 9 * (v20 == 128);
                    }
                  }
                }
              }
            }
          }
        }
        else
        {
          v19 = v18(v4, v93, 0);
        }
        v21 = sub_CA1930((_BYTE *)(v16 + v111[0]));
        v22 = sub_3400BD0(v17, v21, (unsigned int)&v102, v19, 0, 1, 0);
        v24 = v23;
        v25 = v22;
        v26 = (__int64)v113;
        *(_QWORD *)&v113[v16] = v25;
        *(_DWORD *)(v26 + v16 + 8) = v24;
        v16 += 16;
      }
      while ( v16 != 16 * v80 );
      v5 = v93;
      v2 = a1;
      v14 = v113;
    }
    v94 = *(_QWORD *)(v2 + 864);
    v88 = *(_QWORD *)(a2 + 8);
    v27 = sub_AE5020(v5, v88);
    v28 = sub_9208B0(v5, v88);
    v107 = v29;
    v106 = v28;
    v104 = ((1LL << v27) + ((unsigned __int64)(v28 + 7) >> 3) - 1) >> v27 << v27;
    v105 = v29;
    v89 = sub_CA1930(&v104);
    v30 = 1LL << sub_AE5020(v5, *(_QWORD *)(a2 + 8));
    *(_QWORD *)&v31 = sub_33F1270(*(_QWORD *)(v2 + 864), *(_QWORD *)(a2 - 32));
    v32 = *(_QWORD *)(a2 - 32);
    v81 = v31;
    *(_QWORD *)&v82 = sub_338B750(v2, v32);
    *((_QWORD *)&v82 + 1) = v33;
    v37 = sub_33738B0(v2, v32, v33, v34, v35, v36);
    v39 = v37;
    v40 = v38;
    v106 = v102;
    if ( v102 )
    {
      v78 = v38;
      v79 = v37;
      sub_B96E90((__int64)&v106, v102, 1);
      v40 = v78;
      v39 = v79;
    }
    LODWORD(v107) = v103;
    v41 = sub_34118E0(v94, (unsigned int)&v108, (unsigned int)&v106, v39, v40, v30, v82, v81, v89, (__int64)v14);
    v43 = v42;
    v44 = v41;
    if ( v106 )
    {
      v95 = v41;
      sub_B91220((__int64)&v106, v106);
      v41 = v95;
    }
    v45 = *(_QWORD *)(v2 + 864);
    if ( v41 )
    {
      v90 = *(_QWORD *)(v2 + 864);
      v96 = v41;
      nullsub_1875(v41, v90, 0);
      *(_QWORD *)(v90 + 384) = v96;
      *(_DWORD *)(v90 + 392) = v80;
      sub_33E2B60(v90, 0);
    }
    else
    {
      *(_QWORD *)(v45 + 384) = 0;
      *(_DWORD *)(v45 + 392) = v80;
    }
    if ( v113 != v115 )
      _libc_free((unsigned __int64)v113);
    if ( v102 )
      sub_B91220((__int64)&v102, v102);
    if ( (_BYTE *)v111[0] != v112 )
      _libc_free(v111[0]);
    if ( v108 != v110 )
      _libc_free((unsigned __int64)v108);
  }
  v46 = *(_QWORD *)(a2 + 8);
  if ( *(_BYTE *)(v46 + 8) == 14 )
  {
    v99 = *(_QWORD *)(v2 + 864);
    v68 = sub_2E79000(*(__int64 **)(v99 + 40));
    v69 = sub_2D5BAE0(v4, v68, (__int64 *)v46, 0);
    v113 = 0;
    v71 = v99;
    v72 = v69;
    v73 = *(_QWORD *)v2;
    v74 = v70;
    v75 = *(_QWORD *)v2 == 0;
    LODWORD(v114) = *(_DWORD *)(v2 + 848);
    if ( !v75 && &v113 != (_BYTE **)(v73 + 48) )
    {
      v76 = *(_QWORD *)(v73 + 48);
      v113 = (_BYTE *)v76;
      if ( v76 )
      {
        v92 = v70;
        sub_B96E90((__int64)&v113, v76, 1);
        v74 = v92;
        v71 = v99;
      }
    }
    v44 = sub_33FB4C0(v71, v44, v43, &v113, v72, v74);
    v43 = v77;
    if ( v113 )
      sub_B91220((__int64)&v113, (__int64)v113);
  }
  v113 = (_BYTE *)a2;
  result = sub_337DC20(v2 + 8, (__int64 *)&v113);
  *result = v44;
  *((_DWORD *)result + 2) = v43;
  return result;
}
