// Function: sub_6FD310
// Address: 0x6fd310
//
__int64 __fastcall sub_6FD310(int a1, __m128i *a2, __int64 *a3, _DWORD *a4, __int64 *a5, _BYTE *a6)
{
  __m128i *v6; // r12
  __int64 i; // r15
  __int64 j; // r14
  __int64 v9; // rdi
  int v10; // ebx
  int v11; // eax
  __int64 v12; // r8
  unsigned int v13; // r10d
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 m128i_i64; // rdx
  __int64 v19; // rax
  char v21; // al
  __int64 v22; // rdx
  char v23; // al
  __int64 v24; // rax
  int v25; // r10d
  __int64 v26; // rax
  int v27; // eax
  int v28; // r10d
  int v29; // eax
  int v30; // eax
  int v31; // r10d
  __int64 v32; // rax
  char v33; // al
  int v34; // [rsp+8h] [rbp-68h]
  __int64 v35; // [rsp+8h] [rbp-68h]
  __int64 v36; // [rsp+10h] [rbp-60h]
  __int64 v37; // [rsp+18h] [rbp-58h]
  int v42; // [rsp+3Ch] [rbp-34h]
  int v43; // [rsp+3Ch] [rbp-34h]

  v6 = (__m128i *)a3;
  for ( i = a2->m128i_i64[0]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  for ( j = *a3; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  v9 = j;
  v10 = sub_8D2B80(i);
  v11 = sub_8D2B80(j);
  v13 = v11 | v10;
  if ( v11 | v10 )
  {
    if ( v10 )
    {
      if ( v11 )
      {
        if ( *(_QWORD *)(i + 128) != *(_QWORD *)(j + 128) )
        {
          if ( (unsigned int)sub_6E5430() )
          {
            v9 = 1688;
            sub_6851C0(0x698u, a4);
          }
          goto LABEL_28;
        }
        v14 = *(_QWORD *)(i + 160);
        v15 = *(_QWORD *)(j + 160);
        v37 = v14;
        v36 = v15;
        if ( v14 == v15 || (unsigned int)sub_8D97D0(v14, v15, 32, v15, v12) )
          goto LABEL_46;
        if ( (*(_BYTE *)(i + 176) & 1) == 0 && (*(_BYTE *)(j + 176) & 1) == 0 )
          goto LABEL_14;
        v16 = sub_8D4620(i);
        if ( v16 != sub_8D4620(j) )
          goto LABEL_14;
        v21 = *(_BYTE *)(i + 140);
        if ( v21 == 12 )
        {
          v22 = sub_8D4A00(i);
        }
        else
        {
          if ( dword_4F077C0 && (v21 == 1 || v21 == 7) )
          {
            v33 = *(_BYTE *)(j + 140);
            if ( v33 != 12 )
            {
              if ( v33 == 7 )
                goto LABEL_46;
              v22 = 1;
              if ( v33 == 1 )
                goto LABEL_46;
              goto LABEL_44;
            }
            v22 = 1;
LABEL_79:
            v35 = v22;
            v24 = sub_8D4A00(j);
            v22 = v35;
LABEL_45:
            if ( v24 != v22 )
            {
LABEL_14:
              v9 = v37;
              if ( !(unsigned int)sub_8D2780(v37)
                || (v9 = v36, !(unsigned int)sub_8D2780(v36))
                || (v9 = v37, (v34 = sub_8D7480(v37, v36)) == 0) )
              {
                if ( (unsigned int)sub_6E5430() )
                {
                  v9 = 1694;
                  sub_6851C0(0x69Eu, a4);
                }
                goto LABEL_28;
              }
              v25 = (unsigned __int16)a1;
              if ( (unsigned __int16)(a1 - 33) > 0x21u
                || (v32 = 0x380060001LL, !_bittest64(&v32, (unsigned int)(a1 - 33))) )
              {
                *a5 = i;
                *a6 = sub_6E9930((unsigned __int16)a1, i);
LABEL_59:
                if ( (unsigned int)sub_8D27E0(v37) )
                {
                  v9 = (unsigned __int8)*a6;
                  if ( !(unsigned int)sub_730030(v9) )
                  {
                    *a5 = j;
                    sub_6FC3F0(j, a2, 1u);
                    goto LABEL_29;
                  }
                  if ( (unsigned int)sub_6E5430() )
                  {
                    v9 = 1694;
                    sub_6851C0(0x69Eu, a4);
                  }
                  goto LABEL_28;
                }
LABEL_49:
                sub_6FC3F0(i, v6, 1u);
                goto LABEL_29;
              }
LABEL_53:
              v42 = v25;
              v27 = sub_8D2930(v37);
              v28 = v42;
              if ( !v27 )
              {
                v9 = v37;
                v29 = sub_8D3D40(v37);
                v28 = v42;
                if ( !v29 )
                {
LABEL_55:
                  if ( (unsigned int)sub_6E5430() )
                  {
                    v9 = 1695;
                    sub_6851C0(0x69Fu, a4);
                  }
LABEL_28:
                  *a5 = sub_72C930(v9);
                  *a6 = 119;
LABEL_29:
                  v13 = 1;
                  if ( qword_4F04C50 )
                  {
                    v19 = *(_QWORD *)(qword_4F04C50 + 32LL);
                    if ( v19 )
                    {
                      if ( (*(_BYTE *)(v19 + 198) & 0x10) != 0 )
                      {
                        sub_6851C0(0xE6Eu, a4);
                        return 1;
                      }
                    }
                  }
                  return v13;
                }
              }
              v43 = v28;
              v30 = sub_8D2930(v36);
              v31 = v43;
              if ( !v30 )
              {
                v9 = v36;
                if ( !(unsigned int)sub_8D3D40(v36) )
                  goto LABEL_55;
                v31 = v43;
              }
              *a5 = i;
              *a6 = sub_6E9930(v31, i);
              if ( !v34 )
                goto LABEL_49;
              goto LABEL_59;
            }
LABEL_46:
            v25 = (unsigned __int16)a1;
            if ( (unsigned __int16)(a1 - 33) > 0x21u
              || (v26 = 0x380060001LL, !_bittest64(&v26, (unsigned int)(a1 - 33))) )
            {
              *a5 = i;
              *a6 = sub_6E9930((unsigned __int16)a1, i);
              goto LABEL_49;
            }
            v34 = 0;
            goto LABEL_53;
          }
          v22 = *(_QWORD *)(i + 128);
        }
        v23 = *(_BYTE *)(j + 140);
        if ( v23 != 12 )
        {
          if ( dword_4F077C0 && (v23 == 1 || v23 == 7) )
          {
            v24 = 1;
            goto LABEL_45;
          }
LABEL_44:
          v24 = *(_QWORD *)(j + 128);
          goto LABEL_45;
        }
        goto LABEL_79;
      }
    }
    else
    {
      v17 = i;
      v6 = a2;
      i = j;
      j = v17;
    }
    m128i_i64 = 0;
    if ( v6[1].m128i_i8[0] == 2 )
      m128i_i64 = (__int64)v6[9].m128i_i64;
    if ( (unsigned __int16)(a1 - 52) <= 1u )
    {
      *a5 = i;
      *a6 = sub_6E9930((unsigned __int16)a1, i);
      goto LABEL_29;
    }
    v9 = i;
    if ( (unsigned int)sub_6E8F10(i, j, m128i_i64, (unsigned __int16)(a1 - 41) <= 1u) && v6[1].m128i_i8[1] != 1 )
    {
      sub_6FD210(v6, i);
      *a5 = i;
      *a6 = sub_6E9930((unsigned __int16)a1, i);
      goto LABEL_29;
    }
    if ( (unsigned int)sub_6E5430() )
    {
      v9 = 1687;
      sub_6851C0(0x697u, a4);
    }
    goto LABEL_28;
  }
  return v13;
}
