// Function: sub_10460C0
// Address: 0x10460c0
//
unsigned __int8 *__fastcall sub_10460C0(__int64 a1, unsigned __int8 *a2, __int64 *a3, _DWORD *a4, char a5, char a6)
{
  char v8; // dl
  unsigned __int8 *result; // rax
  int v12; // ecx
  char v13; // dl
  unsigned __int8 *v14; // rax
  int v15; // edx
  _BYTE *v16; // r15
  int v17; // eax
  unsigned __int64 v18; // rax
  __int64 v19; // rcx
  _BYTE *v20; // rsi
  char v21; // al
  char v22; // al
  bool v23; // zf
  unsigned __int8 *v24; // r15
  unsigned __int8 *v25; // rax
  _BYTE *v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rdx
  int v29; // edx
  unsigned __int8 *v30; // r15
  unsigned __int8 v31; // al
  unsigned __int8 *v32; // rdi
  __int64 v33; // rcx
  __int64 v34; // rdx
  unsigned __int8 *v35; // r8
  char v36; // al
  unsigned __int8 *v37; // r8
  __int64 v38; // rdi
  __int64 v39; // rax
  unsigned __int8 v40; // al
  __int64 v41; // rax
  __int64 v42; // rdi
  int v43; // eax
  int v44; // esi
  unsigned int v45; // ecx
  __int64 *v46; // rax
  __int64 v47; // r8
  __int64 v48; // rsi
  __int64 v49; // rcx
  int v50; // edx
  __int64 v51; // rdx
  __int64 v52; // rdx
  unsigned __int8 *v53; // rax
  char v54; // al
  unsigned __int8 *v55; // rax
  unsigned __int8 *v56; // rcx
  __int64 v57; // rdx
  __int64 v58; // rdx
  __int64 v59; // rdx
  int v60; // edx
  int v61; // esi
  __int64 v62; // rdx
  __int64 v63; // rdx
  int v64; // eax
  int v65; // r9d
  unsigned __int8 *v66; // [rsp+0h] [rbp-F0h]
  __int64 v67; // [rsp+8h] [rbp-E8h]
  __int64 v68; // [rsp+10h] [rbp-E0h]
  __int64 v69; // [rsp+10h] [rbp-E0h]
  __int64 v70; // [rsp+18h] [rbp-D8h]
  char v71; // [rsp+20h] [rbp-D0h]
  char v72; // [rsp+20h] [rbp-D0h]
  __int64 v73; // [rsp+20h] [rbp-D0h]
  unsigned __int8 *v74; // [rsp+20h] [rbp-D0h]
  _BYTE *v76; // [rsp+28h] [rbp-C8h]
  __m128i v77; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v78; // [rsp+40h] [rbp-B0h]
  __int64 v79; // [rsp+48h] [rbp-A8h]
  __int64 v80; // [rsp+50h] [rbp-A0h]
  __int64 v81; // [rsp+58h] [rbp-98h]
  char v82[8]; // [rsp+70h] [rbp-80h] BYREF
  __m128i v83; // [rsp+78h] [rbp-78h]
  __int64 v84; // [rsp+88h] [rbp-68h]
  __int64 v85; // [rsp+90h] [rbp-60h]
  __int64 v86; // [rsp+98h] [rbp-58h]
  __int64 v87; // [rsp+A0h] [rbp-50h]
  _BYTE *v88; // [rsp+A8h] [rbp-48h]
  unsigned __int8 *v89; // [rsp+B0h] [rbp-40h]
  char v90; // [rsp+B8h] [rbp-38h]

  v8 = *a2;
  if ( (unsigned int)*a2 - 26 > 1 )
    return a2;
  if ( a6 )
  {
    v30 = (unsigned __int8 *)*((_QWORD *)a2 + 9);
    if ( (v30[7] & 0x20) != 0 )
    {
      v67 = *(_QWORD *)(*(_QWORD *)(a1 + 2392) + 8LL);
      if ( sub_B91C10((__int64)v30, 16) && !sub_B46560(v30) )
      {
        v31 = *v30;
        v32 = 0;
        if ( *v30 > 0x1Cu && (v31 == 61 || v31 == 62) )
          v32 = (unsigned __int8 *)*((_QWORD *)v30 - 4);
        v66 = sub_BD3990(v32, 16);
        if ( *v66 > 0x15u )
        {
          v33 = *((_QWORD *)v66 + 2);
          if ( v33 )
          {
            v34 = (__int64)v30;
            do
            {
              v35 = *(unsigned __int8 **)(v33 + 24);
              if ( *v35 > 0x1Cu && v30 != v35 )
              {
                v70 = v33;
                v73 = v34;
                v68 = *(_QWORD *)(v33 + 24);
                v36 = sub_B19DB0(v67, v68, v34);
                v34 = v73;
                v33 = v70;
                if ( v36 )
                {
                  v37 = (unsigned __int8 *)v68;
                  if ( (*(_BYTE *)(v68 + 7) & 0x20) != 0 )
                  {
                    v38 = v68;
                    v69 = v73;
                    v74 = v37;
                    v39 = sub_B91C10(v38, 16);
                    v33 = v70;
                    v34 = v69;
                    if ( v39 )
                    {
                      v40 = *v74;
                      if ( *v74 > 0x1Cu && (v40 == 61 || v40 == 62) )
                      {
                        v53 = (unsigned __int8 *)*((_QWORD *)v74 - 4);
                        if ( v53 )
                        {
                          if ( v66 == v53 )
                          {
                            v54 = sub_B46560(v74);
                            v34 = v69;
                            v33 = v70;
                            if ( !v54 )
                              v34 = (__int64)v74;
                          }
                        }
                      }
                    }
                  }
                }
              }
              v33 = *(_QWORD *)(v33 + 8);
            }
            while ( v33 );
            if ( v30 != (unsigned __int8 *)v34 )
            {
              v41 = *(_QWORD *)(a1 + 2392);
              v42 = *(_QWORD *)(v41 + 40);
              v43 = *(_DWORD *)(v41 + 56);
              if ( v43 )
              {
                v44 = v43 - 1;
                v45 = (v43 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
                v46 = (__int64 *)(v42 + 16LL * v45);
                v47 = *v46;
                if ( v34 == *v46 )
                {
LABEL_63:
                  result = (unsigned __int8 *)v46[1];
                  if ( *result == 26 )
                    return (unsigned __int8 *)*((_QWORD *)result - 4);
                  return result;
                }
                v64 = 1;
                while ( v47 != -4096 )
                {
                  v65 = v64 + 1;
                  v45 = v44 & (v64 + v45);
                  v46 = (__int64 *)(v42 + 16LL * v45);
                  v47 = *v46;
                  if ( v34 == *v46 )
                    goto LABEL_63;
                  v64 = v65;
                }
              }
              BUG();
            }
          }
        }
      }
      v8 = *a2;
    }
  }
  if ( v8 != 27 )
  {
    v14 = a2 - 64;
    if ( v8 == 26 )
      v14 = a2 - 32;
    result = *(unsigned __int8 **)v14;
    if ( result )
    {
      v15 = *result == 27 ? *((_DWORD *)result + 20) : *((_DWORD *)result + 18);
      if ( *((_DWORD *)a2 + 20) == v15 )
        return result;
    }
    goto LABEL_18;
  }
  result = (unsigned __int8 *)*((_QWORD *)a2 - 4);
  if ( !result
    || (*result != 27 ? (v12 = *((_DWORD *)result + 18)) : (v12 = *((_DWORD *)result + 20)), *((_DWORD *)a2 + 21) != v12) )
  {
LABEL_18:
    v13 = 0;
    goto LABEL_19;
  }
  if ( !a5 )
    return result;
  v13 = a5;
LABEL_19:
  v16 = (_BYTE *)*((_QWORD *)a2 + 9);
  v17 = (unsigned __int8)*v16;
  if ( (unsigned __int8)(v17 - 34) <= 0x33u )
  {
    v48 = 0x8000000000041LL;
    if ( _bittest64(&v48, (unsigned int)(v17 - 34)) )
      goto LABEL_22;
    v18 = (unsigned int)(v17 - 29);
LABEL_21:
    v19 = 0x110000800000220LL;
    if ( !_bittest64(&v19, v18) )
      goto LABEL_22;
    return a2;
  }
  v18 = (unsigned int)(v17 - 29);
  if ( (unsigned int)v18 <= 0x38 )
    goto LABEL_21;
LABEL_22:
  if ( (unsigned __int8)(*v16 - 34) <= 0x33u
    && (v49 = 0x8000000000041LL, _bittest64(&v49, (unsigned int)(unsigned __int8)*v16 - 34)) )
  {
    v82[0] = 1;
    v83.m128i_i64[0] = 0;
    v83.m128i_i64[1] = -1;
    v84 = 0;
    v85 = 0;
    v86 = 0;
    v87 = 0;
    v88 = v16;
    v89 = a2;
    v90 = 0;
  }
  else
  {
    v20 = (_BYTE *)*((_QWORD *)a2 + 9);
    v82[0] = 0;
    v71 = v13;
    v83.m128i_i64[0] = 0;
    v83.m128i_i64[1] = -1;
    v84 = 0;
    v85 = 0;
    v86 = 0;
    v87 = 0;
    v88 = v16;
    v89 = a2;
    v90 = 0;
    sub_D66840(&v77, v20);
    v13 = v71;
    v83 = v77;
    v84 = v78;
    v85 = v79;
    v86 = v80;
    v87 = v81;
  }
  if ( *v16 == 61 && (v72 = v13, v21 = sub_103ADB0(a3, (__int64)v16), v13 = v72, v21) )
  {
    v56 = a2 - 32;
    v57 = *((_QWORD *)a2 - 4);
    result = *(unsigned __int8 **)(*(_QWORD *)(a1 + 2392) + 128LL);
    if ( *a2 == 27 )
    {
      if ( v57 )
      {
        v58 = *((_QWORD *)a2 - 3);
        **((_QWORD **)a2 - 2) = v58;
        if ( v58 )
          *(_QWORD *)(v58 + 16) = *((_QWORD *)a2 - 2);
      }
      *((_QWORD *)a2 - 4) = result;
      if ( result )
      {
        v59 = *((_QWORD *)result + 2);
        *((_QWORD *)a2 - 3) = v59;
        if ( v59 )
          *(_QWORD *)(v59 + 16) = a2 - 24;
        *((_QWORD *)a2 - 2) = result + 16;
        *((_QWORD *)result + 2) = v56;
      }
      if ( *result == 27 )
        v60 = *((_DWORD *)result + 20);
      else
        v60 = *((_DWORD *)result + 18);
      *((_DWORD *)a2 + 21) = v60;
    }
    else
    {
      if ( *result == 27 )
        v61 = *((_DWORD *)result + 20);
      else
        v61 = *((_DWORD *)result + 18);
      *((_DWORD *)a2 + 20) = v61;
      if ( v57 )
      {
        v62 = *((_QWORD *)a2 - 3);
        **((_QWORD **)a2 - 2) = v62;
        if ( v62 )
          *(_QWORD *)(v62 + 16) = *((_QWORD *)a2 - 2);
      }
      *((_QWORD *)a2 - 4) = result;
      v63 = *((_QWORD *)result + 2);
      *((_QWORD *)a2 - 3) = v63;
      if ( v63 )
        *(_QWORD *)(v63 + 16) = a2 - 24;
      *((_QWORD *)a2 - 2) = result + 16;
      *((_QWORD *)result + 2) = v56;
    }
  }
  else
  {
    v22 = *a2;
    if ( v13 )
    {
      if ( v22 == 27 )
      {
        result = (unsigned __int8 *)*((_QWORD *)a2 - 4);
      }
      else
      {
        v23 = v22 == 26;
        v55 = a2 - 32;
        if ( !v23 )
          v55 = a2 - 64;
        result = *(unsigned __int8 **)v55;
      }
    }
    else
    {
      v23 = v22 == 26;
      v24 = a2 - 32;
      v25 = a2 - 64;
      if ( v23 )
        v25 = a2 - 32;
      v26 = *(_BYTE **)v25;
      if ( v26 == *(_BYTE **)(*(_QWORD *)(a1 + 2392) + 128LL) )
      {
        v76 = v26;
        sub_103BA70((__int64)a2, (__int64)v26);
        return v76;
      }
      result = sub_1044BF0(a1, (__int64)a3, v26, (__int64)v82, a4);
      if ( *a2 == 27 )
      {
        if ( *((_QWORD *)a2 - 4) )
        {
          v27 = *((_QWORD *)a2 - 3);
          **((_QWORD **)a2 - 2) = v27;
          if ( v27 )
            *(_QWORD *)(v27 + 16) = *((_QWORD *)a2 - 2);
        }
        *((_QWORD *)a2 - 4) = result;
        if ( result )
        {
          v28 = *((_QWORD *)result + 2);
          *((_QWORD *)a2 - 3) = v28;
          if ( v28 )
            *(_QWORD *)(v28 + 16) = a2 - 24;
          *((_QWORD *)a2 - 2) = result + 16;
          *((_QWORD *)result + 2) = v24;
        }
        if ( *result == 27 )
          v29 = *((_DWORD *)result + 20);
        else
          v29 = *((_DWORD *)result + 18);
        *((_DWORD *)a2 + 21) = v29;
      }
      else
      {
        if ( *result == 27 )
          v50 = *((_DWORD *)result + 20);
        else
          v50 = *((_DWORD *)result + 18);
        v23 = *((_QWORD *)a2 - 4) == 0;
        *((_DWORD *)a2 + 20) = v50;
        if ( !v23 )
        {
          v51 = *((_QWORD *)a2 - 3);
          **((_QWORD **)a2 - 2) = v51;
          if ( v51 )
            *(_QWORD *)(v51 + 16) = *((_QWORD *)a2 - 2);
        }
        *((_QWORD *)a2 - 4) = result;
        v52 = *((_QWORD *)result + 2);
        *((_QWORD *)a2 - 3) = v52;
        if ( v52 )
          *(_QWORD *)(v52 + 16) = a2 - 24;
        *((_QWORD *)a2 - 2) = result + 16;
        *((_QWORD *)result + 2) = v24;
      }
    }
    if ( a5 && *result == 28 && *a2 == 27 )
    {
      if ( *a4 )
      {
        v90 = 1;
        return sub_1044BF0(a1, (__int64)a3, result, (__int64)v82, a4);
      }
    }
  }
  return result;
}
