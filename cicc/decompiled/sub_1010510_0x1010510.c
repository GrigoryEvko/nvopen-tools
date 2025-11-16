// Function: sub_1010510
// Address: 0x1010510
//
__int64 __fastcall sub_1010510(char *a1, unsigned __int64 **a2, const __m128i *a3, int a4)
{
  char v8; // al
  __int64 v9; // rdx
  __int64 result; // rax
  char v11; // al
  __int64 *v12; // rax
  unsigned __int8 *v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // r15
  bool v16; // zf
  _BYTE *v17; // rdx
  unsigned __int64 *v18; // rsi
  unsigned __int64 **v19; // r15
  _BYTE *v20; // r15
  char v21; // al
  __int64 v22; // rsi
  __int64 *v23; // rdx
  __int64 *v24; // rcx
  char v25; // al
  unsigned __int64 **v26; // rsi
  __int64 v27; // rcx
  unsigned int v28; // r15d
  bool v29; // al
  char v30; // al
  __int64 v31; // rax
  _BYTE *v32; // rax
  char v33; // r8
  __int64 v34; // r15
  _BYTE *v35; // rax
  unsigned __int8 *v36; // rcx
  char v37; // dl
  unsigned int v38; // r15d
  unsigned int v39; // r15d
  __int64 v40; // rax
  __int64 v41; // [rsp-10h] [rbp-A0h]
  int v42; // [rsp+Ch] [rbp-84h]
  unsigned __int8 *v43; // [rsp+10h] [rbp-80h]
  char v44; // [rsp+10h] [rbp-80h]
  int v45; // [rsp+10h] [rbp-80h]
  unsigned __int8 *v46; // [rsp+18h] [rbp-78h]
  __int64 *v47; // [rsp+28h] [rbp-68h] BYREF
  __int64 *v48; // [rsp+30h] [rbp-60h] BYREF
  unsigned __int64 *v49; // [rsp+38h] [rbp-58h] BYREF
  __int64 *v50; // [rsp+40h] [rbp-50h] BYREF
  unsigned __int64 **v51; // [rsp+48h] [rbp-48h] BYREF
  __int64 *v52; // [rsp+50h] [rbp-40h]
  _QWORD *v53; // [rsp+58h] [rbp-38h] BYREF

  v8 = *a1;
  v50 = 0;
  v51 = a2;
  if ( v8 == 59 )
  {
    v25 = sub_995B10(&v50, *((_QWORD *)a1 - 8));
    v26 = (unsigned __int64 **)*((_QWORD *)a1 - 4);
    if ( v25 && v26 == v51 || (unsigned __int8)sub_995B10(&v50, (__int64)v26) && *((unsigned __int64 ***)a1 - 8) == v51 )
      return sub_AD6530(*((_QWORD *)a1 + 1), (__int64)v26);
    v8 = *a1;
  }
  if ( v8 != 58 )
  {
LABEL_3:
    v9 = (unsigned int)sub_104A6F0(a1, a2, 1);
    result = (__int64)a2;
    if ( (_BYTE)v9 )
      return result;
    v11 = *a1;
    if ( *a1 != 44 )
    {
LABEL_6:
      v50 = (__int64 *)a2;
      v51 = 0;
      if ( v11 == 42 )
      {
        if ( a2 != *((unsigned __int64 ***)a1 - 8) )
          return sub_1010290(0x1Cu, a1, (unsigned __int8 *)a2, a3, a4);
        if ( (unsigned __int8)sub_995B10(&v51, *((_QWORD *)a1 - 4))
          && (unsigned __int8)sub_9B64A0(
                                (__int64)a2,
                                a3->m128i_i64[0],
                                1u,
                                0,
                                a3[2].m128i_i64[0],
                                a3[2].m128i_i64[1],
                                a3[1].m128i_i64[1],
                                1) )
        {
          return sub_AD6530((__int64)a2[1], v41);
        }
        v11 = *a1;
      }
      if ( v11 == 54 )
      {
        v12 = (__int64 *)*((_QWORD *)a1 - 8);
        if ( v12 )
        {
          v13 = (unsigned __int8 *)*((_QWORD *)a1 - 4);
          v47 = (__int64 *)*((_QWORD *)a1 - 8);
          v14 = *v13;
          v15 = (__int64)(v13 + 24);
          if ( (_BYTE)v14 == 17 )
          {
LABEL_10:
            v16 = *(_BYTE *)a2 == 42;
            v50 = v12;
            v51 = &v49;
            LOBYTE(v52) = 0;
            v53 = 0;
            if ( v16 )
            {
              v17 = *(a2 - 8);
              if ( *v17 == 54 && v12 == *((__int64 **)v17 - 8) )
              {
                if ( (unsigned __int8)sub_991580((__int64)&v51, *((_QWORD *)v17 - 4)) )
                {
                  if ( (unsigned __int8)sub_995B10(&v53, (__int64)*(a2 - 4)) )
                  {
                    if ( (unsigned __int8)sub_9B64A0(
                                            (__int64)v47,
                                            a3->m128i_i64[0],
                                            1u,
                                            0,
                                            a3[2].m128i_i64[0],
                                            a3[2].m128i_i64[1],
                                            0,
                                            1) )
                    {
                      v18 = v49;
                      if ( (int)sub_C49970(v15, v49) >= 0 )
                        return sub_AD6530(*((_QWORD *)a1 + 1), (__int64)v18);
                    }
                  }
                }
              }
            }
            return sub_1010290(0x1Cu, a1, (unsigned __int8 *)a2, a3, a4);
          }
          if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v13 + 1) + 8LL) - 17 <= 1 && (unsigned __int8)v14 <= 0x15u )
          {
            v32 = sub_AD7630((__int64)v13, 0, v14);
            if ( v32 )
            {
              if ( *v32 == 17 )
              {
                v15 = (__int64)(v32 + 24);
                v12 = v47;
                goto LABEL_10;
              }
            }
          }
        }
      }
      return sub_1010290(0x1Cu, a1, (unsigned __int8 *)a2, a3, a4);
    }
    v27 = *((_QWORD *)a1 - 8);
    if ( *(_BYTE *)v27 == 17 )
    {
      v28 = *(_DWORD *)(v27 + 32);
      if ( v28 <= 0x40 )
        v29 = *(_QWORD *)(v27 + 24) == 0;
      else
        v29 = v28 == (unsigned int)sub_C444A0(v27 + 24);
    }
    else
    {
      v34 = *(_QWORD *)(v27 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v34 + 8) - 17 > 1 || *(_BYTE *)v27 > 0x15u )
        return sub_1010290(0x1Cu, a1, (unsigned __int8 *)a2, a3, a4);
      v43 = (unsigned __int8 *)*((_QWORD *)a1 - 8);
      v35 = sub_AD7630(v27, 0, v9);
      v36 = v43;
      v37 = 0;
      if ( !v35 || *v35 != 17 )
      {
        if ( *(_BYTE *)(v34 + 8) == 17 )
        {
          v42 = *(_DWORD *)(v34 + 32);
          if ( v42 )
          {
            v39 = 0;
            while ( 1 )
            {
              v44 = v37;
              v46 = v36;
              v40 = sub_AD69F0(v36, v39);
              v36 = v46;
              v37 = v44;
              if ( !v40 )
                break;
              if ( *(_BYTE *)v40 != 13 )
              {
                if ( *(_BYTE *)v40 != 17 )
                  goto LABEL_42;
                if ( *(_DWORD *)(v40 + 32) <= 0x40u )
                {
                  if ( *(_QWORD *)(v40 + 24) )
                    goto LABEL_42;
                  v37 = 1;
                }
                else
                {
                  v45 = *(_DWORD *)(v40 + 32);
                  if ( v45 != (unsigned int)sub_C444A0(v40 + 24) )
                    goto LABEL_42;
                  v36 = v46;
                  v37 = 1;
                }
              }
              if ( v42 == ++v39 )
              {
                if ( v37 )
                  goto LABEL_41;
                goto LABEL_42;
              }
            }
          }
        }
        goto LABEL_42;
      }
      v38 = *((_DWORD *)v35 + 8);
      if ( v38 <= 0x40 )
      {
        if ( *((_QWORD *)v35 + 3) )
          goto LABEL_42;
LABEL_41:
        if ( a2 == *((unsigned __int64 ***)a1 - 4) )
        {
          v33 = sub_9B64A0(
                  (__int64)a2,
                  a3->m128i_i64[0],
                  1u,
                  0,
                  a3[2].m128i_i64[0],
                  a3[2].m128i_i64[1],
                  a3[1].m128i_i64[1],
                  1);
          result = (__int64)a2;
          if ( v33 )
            return result;
        }
LABEL_42:
        v11 = *a1;
        goto LABEL_6;
      }
      v29 = v38 == (unsigned int)sub_C444A0((__int64)(v35 + 24));
    }
    if ( !v29 )
      goto LABEL_42;
    goto LABEL_41;
  }
  result = *((_QWORD *)a1 - 8);
  if ( a2 == (unsigned __int64 **)result )
    return result;
  v19 = (unsigned __int64 **)*((_QWORD *)a1 - 4);
  if ( a2 == v19 )
    return (__int64)a2;
  v51 = 0;
  v50 = (__int64 *)&v47;
  v52 = (__int64 *)&v48;
  if ( !result )
  {
LABEL_73:
    if ( !v19 )
      goto LABEL_3;
    goto LABEL_22;
  }
  v47 = (__int64 *)result;
  if ( *(_BYTE *)v19 != 59 )
  {
LABEL_22:
    *v50 = (__int64)v19;
    v20 = (_BYTE *)*((_QWORD *)a1 - 8);
    if ( *v20 != 59 )
      goto LABEL_3;
    v21 = sub_995B10(&v51, *((_QWORD *)v20 - 8));
    v22 = *((_QWORD *)v20 - 4);
    if ( v21 && v22 )
      goto LABEL_25;
    if ( !(unsigned __int8)sub_995B10(&v51, v22) )
      goto LABEL_3;
    v31 = *((_QWORD *)v20 - 8);
    if ( !v31 )
      goto LABEL_3;
LABEL_48:
    *v52 = v31;
    goto LABEL_26;
  }
  v30 = sub_995B10(&v51, (__int64)*(v19 - 8));
  v22 = (__int64)*(v19 - 4);
  if ( !v30 || !v22 )
  {
    if ( (unsigned __int8)sub_995B10(&v51, v22) )
    {
      v31 = (__int64)*(v19 - 8);
      if ( v31 )
        goto LABEL_48;
    }
    v19 = (unsigned __int64 **)*((_QWORD *)a1 - 4);
    goto LABEL_73;
  }
LABEL_25:
  *v52 = v22;
LABEL_26:
  if ( *(_BYTE *)a2 != 58 )
    goto LABEL_3;
  result = (__int64)v47;
  v23 = (__int64 *)*(a2 - 8);
  v24 = (__int64 *)*(a2 - 4);
  if ( (v47 != v23 || v24 != v48) && (v24 != v47 || v48 != v23) )
    goto LABEL_3;
  return result;
}
