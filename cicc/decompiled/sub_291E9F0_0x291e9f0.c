// Function: sub_291E9F0
// Address: 0x291e9f0
//
__int64 __fastcall sub_291E9F0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r14
  bool v3; // r8
  __int64 result; // rax
  unsigned __int8 v5; // cl
  char v6; // al
  __int64 v7; // r12
  char v8; // bl
  __int64 v9; // r15
  _QWORD *v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // rbx
  __int64 v15; // r9
  __int64 v16; // r12
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  const char *v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 *v25; // rdx
  unsigned __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 *v29; // rax
  _QWORD *v30; // r15
  __int64 *v31; // rbx
  __int64 *v32; // r14
  __int64 v33; // rax
  _QWORD *v34; // rsi
  _QWORD *v35; // rdx
  _QWORD *v36; // rax
  __int64 v37; // rcx
  __int64 *v38; // rax
  __int64 v39; // [rsp+8h] [rbp-1C8h]
  __int64 v40; // [rsp+10h] [rbp-1C0h]
  _QWORD *v41; // [rsp+10h] [rbp-1C0h]
  __int64 v42; // [rsp+18h] [rbp-1B8h]
  __int64 v43; // [rsp+30h] [rbp-1A0h]
  __int64 v45; // [rsp+40h] [rbp-190h]
  __int64 *v46; // [rsp+40h] [rbp-190h]
  __int64 v47; // [rsp+48h] [rbp-188h]
  __int64 *v48; // [rsp+48h] [rbp-188h]
  __int64 v49; // [rsp+58h] [rbp-178h] BYREF
  __int64 v50; // [rsp+60h] [rbp-170h] BYREF
  unsigned int v51; // [rsp+68h] [rbp-168h]
  int v52; // [rsp+6Ch] [rbp-164h]
  __int64 v53[4]; // [rsp+70h] [rbp-160h] BYREF
  _QWORD v54[4]; // [rsp+90h] [rbp-140h] BYREF
  __int16 v55; // [rsp+B0h] [rbp-120h]
  __int64 v56; // [rsp+C0h] [rbp-110h] BYREF
  _BYTE *v57; // [rsp+C8h] [rbp-108h]
  __int64 v58; // [rsp+D0h] [rbp-100h]
  _BYTE v59[16]; // [rsp+D8h] [rbp-F8h] BYREF
  _QWORD *v60; // [rsp+E8h] [rbp-E8h]
  __int64 v61; // [rsp+F0h] [rbp-E0h]
  _QWORD v62[6]; // [rsp+F8h] [rbp-D8h] BYREF
  char v63; // [rsp+128h] [rbp-A8h]
  __int64 v64; // [rsp+130h] [rbp-A0h]
  __int64 v65; // [rsp+138h] [rbp-98h]
  __int64 v66; // [rsp+140h] [rbp-90h]
  __int64 v67; // [rsp+148h] [rbp-88h]
  __int64 v68; // [rsp+150h] [rbp-80h]
  __int64 *v69; // [rsp+158h] [rbp-78h]
  __int64 v70; // [rsp+160h] [rbp-70h]
  _BYTE v71[32]; // [rsp+168h] [rbp-68h] BYREF
  __int64 *v72; // [rsp+188h] [rbp-48h] BYREF
  __int64 i; // [rsp+190h] [rbp-40h]
  _BYTE v74[56]; // [rsp+198h] [rbp-38h] BYREF

  v2 = (_QWORD *)a2;
  v3 = sub_B46500((unsigned __int8 *)a2);
  result = 0;
  if ( !v3 )
  {
    result = *(_BYTE *)(a2 + 2) & 1;
    if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
    {
      return 0;
    }
    else
    {
      v5 = *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL);
      if ( v5 > 3u && v5 != 5 )
      {
        if ( v5 > 0x14u || (result = ((0x165450uLL >> v5) & 1) == 0, ((0x165450uLL >> v5) & 1) == 0) )
        {
          v6 = sub_2912520(a2, 0);
          v7 = *(_QWORD *)(a1 + 192);
          v8 = v6;
          v9 = *(_QWORD *)(a1 + 184);
          sub_B91FC0(v53, a2);
          v39 = *(_QWORD *)(a2 + 8);
          v10 = *(_QWORD **)(v7 + 72);
          v42 = v53[1];
          v47 = v53[0];
          v43 = v53[2];
          v45 = v53[3];
          v40 = **(_QWORD **)(a1 + 176);
          v58 = 0x400000000LL;
          v56 = v7;
          v57 = v59;
          v11 = sub_BCB2D0(v10);
          v12 = sub_ACD640(v11, 0, 0);
          v61 = 0x400000001LL;
          v62[0] = v12;
          v63 = v8;
          v60 = v62;
          v62[4] = v40;
          v62[5] = v39;
          v64 = v9;
          sub_D5F1F0(v7, a2);
          v14 = *(_QWORD *)(a2 + 16);
          v70 = 0x400000000LL;
          v15 = v42;
          v65 = v47;
          v69 = (__int64 *)v71;
          v72 = (__int64 *)v74;
          v66 = v42;
          v67 = v43;
          v68 = v45;
          for ( i = 0x100000000LL; v14; v14 = *(_QWORD *)(v14 + 8) )
          {
            while ( 1 )
            {
              v16 = *(_QWORD *)(v14 + 24);
              if ( *(_BYTE *)v16 == 85 )
              {
                v17 = *(_QWORD *)(v16 - 32);
                if ( v17 )
                {
                  if ( !*(_BYTE *)v17
                    && *(_QWORD *)(v17 + 24) == *(_QWORD *)(v16 + 80)
                    && (*(_BYTE *)(v17 + 33) & 0x20) != 0
                    && *(_DWORD *)(v17 + 36) == 171 )
                  {
                    break;
                  }
                }
              }
              v14 = *(_QWORD *)(v14 + 8);
              if ( !v14 )
                goto LABEL_19;
            }
            v18 = (unsigned int)i;
            v19 = (unsigned int)i + 1LL;
            if ( v19 > HIDWORD(i) )
            {
              sub_C8D5F0((__int64)&v72, v74, v19, 8u, v13, v15);
              v18 = (unsigned int)i;
            }
            v72[v18] = v16;
            LODWORD(i) = i + 1;
          }
LABEL_19:
          v49 = sub_ACADE0(*(__int64 ***)(a2 + 8));
          v20 = sub_BD5D20(a2);
          v21 = *(_QWORD *)(a2 + 8);
          v55 = 773;
          v54[0] = v20;
          v54[1] = v22;
          v54[2] = ".fca";
          sub_291B5C0((__int64)&v56, v21, &v49, (__int64)v54, v23, v24);
          v29 = *(__int64 **)(a1 + 176);
          if ( *(_BYTE *)*v29 > 0x1Cu )
            sub_2914720(a1, *v29, v25, v26, v27, v28);
          v48 = v72;
          v46 = &v72[(unsigned int)i];
          if ( v72 != v46 )
          {
            v41 = v2;
            do
            {
              v30 = (_QWORD *)*v48;
              sub_D5F1F0(v56, *v48);
              v31 = &v69[(unsigned int)v70];
              v32 = v69;
              while ( v31 != v32 )
              {
                v33 = *v32;
                v55 = 257;
                ++v32;
                v52 = 0;
                v50 = v33;
                sub_B33D10(v56, 0xABu, 0, 0, (int)&v50, 1, v51, (__int64)v54);
              }
              sub_B43D60(v30);
              ++v48;
            }
            while ( v46 != v48 );
            v2 = v41;
          }
          if ( *(_BYTE *)(a1 + 108) )
          {
            v34 = *(_QWORD **)(a1 + 88);
            v35 = &v34[*(unsigned int *)(a1 + 100)];
            v36 = v34;
            if ( v34 != v35 )
            {
              while ( v2 != (_QWORD *)*v36 )
              {
                if ( v35 == ++v36 )
                  goto LABEL_33;
              }
              v37 = (unsigned int)(*(_DWORD *)(a1 + 100) - 1);
              *(_DWORD *)(a1 + 100) = v37;
              *v36 = v34[v37];
              ++*(_QWORD *)(a1 + 80);
            }
          }
          else
          {
            v38 = sub_C8CA60(a1 + 80, (__int64)v2);
            if ( v38 )
            {
              *v38 = -2;
              ++*(_DWORD *)(a1 + 104);
              ++*(_QWORD *)(a1 + 80);
            }
          }
LABEL_33:
          sub_BD84D0((__int64)v2, v49);
          sub_B43D60(v2);
          if ( v72 != (__int64 *)v74 )
            _libc_free((unsigned __int64)v72);
          if ( v69 != (__int64 *)v71 )
            _libc_free((unsigned __int64)v69);
          if ( v60 != v62 )
            _libc_free((unsigned __int64)v60);
          if ( v57 != v59 )
            _libc_free((unsigned __int64)v57);
          return 1;
        }
      }
    }
  }
  return result;
}
