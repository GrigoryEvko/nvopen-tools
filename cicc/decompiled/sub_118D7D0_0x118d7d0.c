// Function: sub_118D7D0
// Address: 0x118d7d0
//
__int64 __fastcall sub_118D7D0(__int64 a1, const __m128i *a2)
{
  _BYTE *v3; // rdi
  __int64 v4; // rbx
  unsigned int v5; // r15d
  __int64 v6; // rax
  unsigned int v7; // r14d
  unsigned __int8 *v8; // r9
  int v9; // edi
  unsigned __int8 *v10; // rax
  unsigned __int8 *v11; // r9
  void *v12; // rax
  unsigned __int8 *v13; // r9
  unsigned __int8 *v14; // rdx
  bool v15; // al
  void *v16; // rax
  __int64 v17; // rbx
  __int64 *v18; // r15
  __int64 *v20; // rax
  __int64 v21; // r15
  void **v22; // rax
  unsigned __int8 *v23; // rdx
  void *v24; // rax
  _BYTE *v25; // rcx
  __int64 v26; // rdx
  void **v27; // rax
  void **v28; // r15
  void *v29; // rax
  _BYTE *v30; // r15
  __m128i v31; // xmm1
  __int64 v32; // rax
  unsigned __int64 v33; // xmm2_8
  __m128i v34; // xmm3
  unsigned int v35; // r15d
  void **v36; // rax
  void **v37; // rcx
  char v38; // al
  void *v39; // rax
  _BYTE *v40; // rcx
  unsigned int v41; // r15d
  void **v42; // rax
  void **v43; // rdx
  char v44; // al
  void *v45; // rax
  _BYTE *v46; // rdx
  char v47; // [rsp+3h] [rbp-ADh]
  int v48; // [rsp+4h] [rbp-ACh]
  void **v49; // [rsp+8h] [rbp-A8h]
  int v50; // [rsp+8h] [rbp-A8h]
  unsigned __int8 *v51; // [rsp+10h] [rbp-A0h]
  unsigned __int8 *v52; // [rsp+10h] [rbp-A0h]
  unsigned __int8 *v53; // [rsp+10h] [rbp-A0h]
  unsigned __int8 *v54; // [rsp+10h] [rbp-A0h]
  unsigned __int8 *v55; // [rsp+10h] [rbp-A0h]
  unsigned __int8 *v56; // [rsp+10h] [rbp-A0h]
  void **v57; // [rsp+10h] [rbp-A0h]
  unsigned __int8 *v58; // [rsp+18h] [rbp-98h]
  unsigned __int8 *v59; // [rsp+18h] [rbp-98h]
  void **v60; // [rsp+18h] [rbp-98h]
  __int64 v61; // [rsp+18h] [rbp-98h]
  unsigned __int8 *v62; // [rsp+18h] [rbp-98h]
  unsigned __int8 *v63; // [rsp+18h] [rbp-98h]
  unsigned __int8 *v64; // [rsp+18h] [rbp-98h]
  unsigned __int8 *v65; // [rsp+20h] [rbp-90h]
  bool v66; // [rsp+20h] [rbp-90h]
  unsigned __int8 *v67; // [rsp+20h] [rbp-90h]
  __int64 v68; // [rsp+20h] [rbp-90h]
  __int64 *v69; // [rsp+28h] [rbp-88h]
  __m128i v70[2]; // [rsp+30h] [rbp-80h] BYREF
  unsigned __int64 v71; // [rsp+50h] [rbp-60h]
  __int64 v72; // [rsp+58h] [rbp-58h]
  __m128i v73; // [rsp+60h] [rbp-50h]
  __int64 v74; // [rsp+70h] [rbp-40h]

  v3 = *(_BYTE **)(a1 - 96);
  if ( (unsigned __int8)(*v3 - 82) > 1u )
    return 0;
  v69 = (__int64 *)*((_QWORD *)v3 - 8);
  if ( !v69 )
    return 0;
  v4 = *((_QWORD *)v3 - 4);
  if ( *(_BYTE *)v4 > 0x15u )
    return 0;
  v5 = sub_B53900((__int64)v3);
  if ( v5 - 32 <= 1 )
  {
    v6 = -64;
    if ( v5 != 32 )
      v6 = -32;
    v7 = (v5 != 32) + 1;
  }
  else if ( v5 == 1 )
  {
    v7 = 1;
    v6 = -64;
  }
  else
  {
    v6 = -32;
    v7 = 2;
    if ( v5 != 14 )
      return 0;
  }
  v8 = *(unsigned __int8 **)(a1 + v6);
  v9 = *v8;
  if ( (unsigned __int8)v9 <= 0x1Cu || (unsigned int)(v9 - 42) > 0x11 )
    return 0;
  v65 = *(unsigned __int8 **)(a1 + v6);
  v10 = sub_AD93D0(v9 - 29, *((_QWORD *)v8 + 1), 1, 0);
  v11 = v65;
  if ( v10 != (unsigned __int8 *)v4 )
  {
    v66 = v5 > 0xF || v10 == 0;
    if ( v66 )
      return 0;
    if ( *v10 == 18 )
    {
      v51 = v10;
      v58 = v11;
      v12 = sub_C33340();
      v13 = v58;
      v14 = *((void **)v51 + 3) == v12 ? (unsigned __int8 *)*((_QWORD *)v51 + 4) : v51 + 24;
      v15 = (v14[20] & 7) == 3;
    }
    else
    {
      v21 = *((_QWORD *)v10 + 1);
      v59 = v11;
      if ( (unsigned int)*(unsigned __int8 *)(v21 + 8) - 17 > 1 )
        return 0;
      v52 = v10;
      v22 = (void **)sub_AD7630((__int64)v10, 0, (__int64)v10);
      v23 = v52;
      v13 = v59;
      if ( !v22 || (v60 = v22, *(_BYTE *)v22 != 18) )
      {
        if ( *(_BYTE *)(v21 + 8) == 17 )
        {
          v48 = *(_DWORD *)(v21 + 32);
          if ( v48 )
          {
            v47 = 0;
            v35 = 0;
            while ( 1 )
            {
              v55 = v13;
              v62 = v23;
              v36 = (void **)sub_AD69F0(v23, v35);
              v37 = v36;
              if ( !v36 )
                break;
              v38 = *(_BYTE *)v36;
              v49 = v37;
              v23 = v62;
              v13 = v55;
              if ( v38 != 13 )
              {
                v56 = v62;
                v63 = v13;
                if ( v38 != 18 )
                  return 0;
                v39 = sub_C33340();
                v13 = v63;
                v23 = v56;
                v40 = v49[3] == v39 ? v49[4] : v49 + 3;
                if ( (v40[20] & 7) != 3 )
                  return 0;
                v47 = 1;
              }
              if ( v48 == ++v35 )
              {
                if ( v47 )
                  goto LABEL_16;
                return 0;
              }
            }
          }
        }
        return 0;
      }
      v53 = v13;
      v24 = sub_C33340();
      v13 = v53;
      v25 = v60[3] == v24 ? v60[4] : v60 + 3;
      v15 = (v25[20] & 7) == 3;
    }
    if ( !v15 )
      return 0;
LABEL_16:
    if ( *(_BYTE *)v4 == 18 )
    {
      v67 = v13;
      v16 = sub_C33340();
      v11 = v67;
      if ( *(void **)(v4 + 24) == v16 )
        v17 = *(_QWORD *)(v4 + 32);
      else
        v17 = v4 + 24;
      if ( (*(_BYTE *)(v17 + 20) & 7) != 3 )
        return 0;
    }
    else
    {
      v26 = *(_QWORD *)(v4 + 8);
      v54 = v13;
      v61 = v26;
      if ( (unsigned int)*(unsigned __int8 *)(v26 + 8) - 17 > 1 )
        return 0;
      v27 = (void **)sub_AD7630(v4, 0, v26);
      v11 = v54;
      v28 = v27;
      if ( !v27 || *(_BYTE *)v27 != 18 )
      {
        if ( *(_BYTE *)(v61 + 8) == 17 )
        {
          v50 = *(_DWORD *)(v61 + 32);
          if ( v50 )
          {
            v41 = 0;
            while ( 1 )
            {
              v64 = v11;
              v42 = (void **)sub_AD69F0((unsigned __int8 *)v4, v41);
              v43 = v42;
              if ( !v42 )
                break;
              v44 = *(_BYTE *)v42;
              v57 = v43;
              v11 = v64;
              if ( v44 != 13 )
              {
                if ( v44 != 18 )
                  return 0;
                v45 = sub_C33340();
                v11 = v64;
                v46 = v57[3] == v45 ? v57[4] : v57 + 3;
                if ( (v46[20] & 7) != 3 )
                  return 0;
                v66 = 1;
              }
              if ( v50 == ++v41 )
              {
                if ( v66 )
                  goto LABEL_20;
                return 0;
              }
            }
          }
        }
        return 0;
      }
      v29 = sub_C33340();
      v11 = v54;
      v30 = v28[3] == v29 ? v28[4] : v28 + 3;
      if ( (v30[20] & 7) != 3 )
        return 0;
    }
  }
LABEL_20:
  v68 = (__int64)v11;
  if ( sub_B46D50(v11) )
  {
    v18 = *(__int64 **)(v68 - 64);
    v20 = *(__int64 **)(v68 - 32);
    if ( v18 )
    {
      if ( v69 != v20 )
      {
        if ( !v20 || v18 != v69 )
          return 0;
        v18 = *(__int64 **)(v68 - 32);
      }
      goto LABEL_33;
    }
    return 0;
  }
  v18 = *(__int64 **)(v68 - 64);
  if ( !v18 || v69 != *(__int64 **)(v68 - 32) )
    return 0;
LABEL_33:
  if ( (unsigned __int8)sub_920620(v68) )
  {
    if ( !sub_B451E0(v68) )
    {
      v31 = _mm_loadu_si128(a2 + 7);
      v32 = a2[10].m128i_i64[0];
      v33 = _mm_loadu_si128(a2 + 8).m128i_u64[0];
      v70[0] = _mm_loadu_si128(a2 + 6);
      v34 = _mm_loadu_si128(a2 + 9);
      v74 = v32;
      v71 = v33;
      v70[1] = v31;
      v72 = a1;
      v73 = v34;
      if ( (sub_9B4030(v18, 32, 0, v70) & 0x20) != 0 )
        return 0;
    }
  }
  return sub_F20660((__int64)a2, a1, v7, (__int64)v18);
}
