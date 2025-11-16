// Function: sub_6F9840
// Address: 0x6f9840
//
__int64 __fastcall sub_6F9840(const __m128i *a1, unsigned int a2, int a3)
{
  __int64 result; // rax
  __int64 v7; // r8
  int v8; // r13d
  __int64 v9; // r14
  unsigned __int64 v10; // rcx
  char v11; // si
  __int64 *v12; // rdx
  __m128i v13; // xmm1
  __m128i v14; // xmm2
  __m128i v15; // xmm3
  __m128i v16; // xmm4
  __m128i v17; // xmm5
  __m128i v18; // xmm6
  __int8 v19; // al
  __m128i v20; // xmm7
  __m128i v21; // xmm0
  __int64 v22; // rcx
  __int64 v23; // rax
  char j; // dl
  __int64 *v25; // rdi
  __m128i v26; // xmm2
  __m128i v27; // xmm3
  __m128i v28; // xmm4
  __m128i v29; // xmm5
  __m128i v30; // xmm6
  __m128i v31; // xmm7
  __m128i v32; // xmm1
  __m128i v33; // xmm2
  __m128i v34; // xmm3
  __m128i v35; // xmm4
  __m128i v36; // xmm5
  __m128i v37; // xmm6
  __int64 *v38; // rax
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 *v43; // rdi
  __int64 v44; // rax
  _BOOL4 v45; // eax
  int v46; // eax
  __int64 v47; // rcx
  const __m128i *v48; // rsi
  _DWORD *v49; // rdi
  __int64 *v50; // rax
  _DWORD *v51; // rdi
  const __m128i *v52; // rsi
  __int64 i; // rcx
  __int64 *v54; // [rsp+10h] [rbp-330h]
  char v55; // [rsp+20h] [rbp-320h]
  __int64 v56; // [rsp+20h] [rbp-320h]
  __int64 v57; // [rsp+28h] [rbp-318h]
  __int64 v58; // [rsp+28h] [rbp-318h]
  __int64 v59; // [rsp+30h] [rbp-310h]
  __int64 v60; // [rsp+38h] [rbp-308h]
  int v61; // [rsp+4Ch] [rbp-2F4h] BYREF
  _OWORD v62[6]; // [rsp+50h] [rbp-2F0h] BYREF
  __m128i v63; // [rsp+B0h] [rbp-290h]
  __m128i v64; // [rsp+C0h] [rbp-280h]
  __m128i v65; // [rsp+D0h] [rbp-270h]
  _OWORD v66[13]; // [rsp+E0h] [rbp-260h] BYREF
  _BYTE v67[144]; // [rsp+1B0h] [rbp-190h] BYREF
  __int64 v68; // [rsp+240h] [rbp-100h]

  result = a1[1].m128i_u8[1];
  if ( (_BYTE)result == 2 )
  {
    result = sub_8D2600(a1->m128i_i64[0]);
    v8 = result;
    if ( (_DWORD)result )
    {
      result = a1[1].m128i_u8[1];
      if ( (_BYTE)result != 1 )
        return result;
      goto LABEL_37;
    }
    if ( a1[1].m128i_i8[0] != 1 )
      return result;
    v9 = a1[9].m128i_i64[0];
    v61 = 0;
    result = *(unsigned __int8 *)(v9 + 24);
    if ( !dword_4F077BC )
    {
      v59 = 0;
      v60 = 0;
      goto LABEL_45;
    }
    if ( qword_4F077A8 > 0x9C3Fu )
    {
      v59 = 0;
      v60 = 0;
      goto LABEL_45;
    }
    if ( (_BYTE)result != 1 )
    {
      if ( a3 )
        a3 = 0;
      v59 = 0;
      v60 = 0;
      goto LABEL_12;
    }
    if ( *(_BYTE *)(v9 + 56) != 5 )
    {
      v11 = *(_BYTE *)(v9 + 56);
      if ( !a3 )
      {
        v59 = 0;
        v10 = qword_4F077A8;
        v60 = 0;
        if ( qword_4F077A8 > 0x9C3Fu )
          goto LABEL_13;
        goto LABEL_49;
      }
      v59 = 0;
      v60 = 0;
LABEL_48:
      v10 = qword_4F077A8;
      result = *(unsigned __int8 *)(v9 + 24);
      a3 = 0;
      if ( qword_4F077A8 > 0x9C3Fu )
        goto LABEL_13;
      goto LABEL_49;
    }
    v38 = *(__int64 **)(v9 + 72);
    v60 = *(_QWORD *)v9;
    v59 = *v38;
    if ( *(_QWORD *)v9 != *v38 )
    {
      if ( !(unsigned int)sub_8D97D0(*(_QWORD *)v9, *v38, 0, &dword_4F077BC, v7) )
      {
LABEL_105:
        result = *(unsigned __int8 *)(v9 + 24);
LABEL_45:
        if ( a3 )
        {
          a3 = 0;
          if ( (_BYTE)result != 1 )
            goto LABEL_12;
          v11 = *(_BYTE *)(v9 + 56);
          if ( v11 != 5 )
            goto LABEL_48;
          v60 = *(_QWORD *)v9;
          v59 = **(_QWORD **)(v9 + 72);
          a3 = sub_8D2930(v59);
          if ( a3 )
          {
            a3 = sub_8D2930(v60);
            if ( a3 )
            {
              v39 = v59;
              if ( *(_BYTE *)(v59 + 140) == 12 )
              {
                do
                  v39 = *(_QWORD *)(v39 + 160);
                while ( *(_BYTE *)(v39 + 140) == 12 );
              }
              else
              {
                v39 = v59;
              }
              v40 = *(_QWORD *)(v39 + 128);
              v41 = v60;
              if ( *(_BYTE *)(v60 + 140) == 12 )
              {
                do
                  v41 = *(_QWORD *)(v41 + 160);
                while ( *(_BYTE *)(v41 + 140) == 12 );
              }
              else
              {
                v41 = v60;
              }
              a3 = *(_QWORD *)(v41 + 128) == v40;
            }
          }
          result = *(unsigned __int8 *)(v9 + 24);
        }
        if ( (_BYTE)result == 1 )
        {
          v10 = qword_4F077A8;
          v11 = *(_BYTE *)(v9 + 56);
          if ( qword_4F077A8 > 0x9C3Fu && !a3 )
          {
LABEL_13:
            v12 = (__int64 *)&dword_4F077C4;
            if ( dword_4F077C4 != 2 )
            {
LABEL_14:
              result = (__int64)&dword_4F077C0;
              if ( dword_4F077C0 )
              {
LABEL_15:
                if ( qword_4F077A8 <= 0x9C3Fu && *(_BYTE *)(v9 + 24) == 3 )
                {
                  v61 = 1;
                  goto LABEL_18;
                }
              }
LABEL_17:
              if ( !v61 )
                return result;
              goto LABEL_18;
            }
LABEL_63:
            if ( HIDWORD(qword_4F077B4) )
            {
              if ( (_BYTE)result == 1 && v10 <= 0x9E97 && (_DWORD)qword_4F077B4 == 0 && v11 == 94 )
              {
                v42 = *(_QWORD *)(v9 + 72);
                if ( (*(_BYTE *)(v42 + 25) & 1) == 0 )
                {
                  v43 = *(__int64 **)(v9 + 72);
                  v54 = v43;
                  v58 = *(_QWORD *)(v42 + 16);
                  v44 = *(_QWORD *)(v58 + 56);
                  *(_QWORD *)(v42 + 16) = 0;
                  v56 = v44;
                  sub_6E70E0(v43, (__int64)v67);
                  sub_6F97B0((__int64)v67, 0);
                  v12 = v43;
                  if ( v67[17] != 1 || (v45 = sub_6ED0A0((__int64)v67), v12 = v43, v45) )
                  {
                    result = v58;
                    v12[2] = v58;
                  }
                  else
                  {
                    v46 = a1[1].m128i_u8[0];
                    v47 = 36;
                    v48 = a1;
                    v49 = v62;
                    while ( v47 )
                    {
                      *v49 = v48->m128i_i32[0];
                      v48 = (const __m128i *)((char *)v48 + 4);
                      ++v49;
                      --v47;
                    }
                    if ( (_BYTE)v46 == 2 )
                    {
                      v51 = v66;
                      v52 = a1 + 9;
                      for ( i = 52; i; --i )
                      {
                        *v51 = v52->m128i_i32[0];
                        v52 = (const __m128i *)((char *)v52 + 4);
                        ++v51;
                      }
                    }
                    else if ( (_BYTE)v46 == 5 || (_BYTE)v46 == 1 )
                    {
                      *(_QWORD *)&v66[0] = a1[9].m128i_i64[0];
                    }
                    v50 = (__int64 *)sub_73E470(v68, v56, v54);
                    sub_6E7150(v50, (__int64)a1);
                    result = sub_6E4BC0((__int64)a1, (__int64)v62);
                  }
                  goto LABEL_17;
                }
              }
            }
            if ( (unsigned int)sub_8D3A70(a1->m128i_i64[0]) )
            {
              result = sub_6F97B0((__int64)a1, 0);
              goto LABEL_17;
            }
            goto LABEL_14;
          }
LABEL_49:
          if ( v11 == 5 )
          {
            v55 = ((*(_BYTE *)(v9 + 27) >> 1) ^ 1) & 1;
          }
          else
          {
            if ( !dword_4F077C0 )
            {
              v12 = (__int64 *)&dword_4F077C4;
              if ( dword_4F077C4 != 2 )
                goto LABEL_17;
              goto LABEL_63;
            }
            if ( v11 != 103 && v11 != 91 )
            {
              v12 = (__int64 *)&dword_4F077C4;
              if ( dword_4F077C4 != 2 )
                goto LABEL_15;
              goto LABEL_63;
            }
            v55 = 0;
          }
          result = sub_6EED10(v9, &v61, 1, dword_4F077C0, a2, 0);
          if ( a2 )
          {
            if ( !v61 )
              return result;
            if ( v9 == result || !v55 )
            {
              v9 = result;
            }
            else
            {
              v22 = *(_QWORD *)v9;
              v8 = 1;
              v9 = result;
              v60 = v22;
              v59 = *(_QWORD *)result;
            }
LABEL_18:
            v13 = _mm_loadu_si128(a1 + 1);
            v14 = _mm_loadu_si128(a1 + 2);
            v15 = _mm_loadu_si128(a1 + 3);
            v16 = _mm_loadu_si128(a1 + 4);
            v17 = _mm_loadu_si128(a1 + 5);
            v62[0] = _mm_loadu_si128(a1);
            v18 = _mm_loadu_si128(a1 + 6);
            v19 = a1[1].m128i_i8[0];
            v62[1] = v13;
            v20 = _mm_loadu_si128(a1 + 7);
            v62[2] = v14;
            v21 = _mm_loadu_si128(a1 + 8);
            v62[3] = v15;
            v62[4] = v16;
            v62[5] = v17;
            v63 = v18;
            v64 = v20;
            v65 = v21;
            if ( v19 == 2 )
            {
              v26 = _mm_loadu_si128(a1 + 10);
              v27 = _mm_loadu_si128(a1 + 11);
              v28 = _mm_loadu_si128(a1 + 12);
              v29 = _mm_loadu_si128(a1 + 13);
              v66[0] = _mm_loadu_si128(a1 + 9);
              v30 = _mm_loadu_si128(a1 + 14);
              v31 = _mm_loadu_si128(a1 + 15);
              v66[1] = v26;
              v32 = _mm_loadu_si128(a1 + 16);
              v33 = _mm_loadu_si128(a1 + 17);
              v66[2] = v27;
              v34 = _mm_loadu_si128(a1 + 18);
              v66[3] = v28;
              v35 = _mm_loadu_si128(a1 + 19);
              v66[4] = v29;
              v36 = _mm_loadu_si128(a1 + 20);
              v66[5] = v30;
              v37 = _mm_loadu_si128(a1 + 21);
              v66[6] = v31;
              v66[7] = v32;
              v66[8] = v33;
              v66[9] = v34;
              v66[10] = v35;
              v66[11] = v36;
              v66[12] = v37;
            }
            else
            {
              if ( v19 != 5 && v19 != 1 )
              {
                v57 = 0;
                if ( !v8 )
                  goto LABEL_33;
LABEL_22:
                if ( dword_4F077C0 && (unsigned int)sub_6EEB90(v59, v60, (__int64)v12) )
                {
                  v57 = v60;
                  if ( !dword_4F077BC || a3 )
                    goto LABEL_84;
                }
                else
                {
                  if ( !dword_4F077BC || a3 )
                    goto LABEL_89;
                  v57 = 0;
                }
                if ( qword_4F077A8 > 0x76BFu
                  || (!(unsigned int)sub_8D2780(v60) || !(unsigned int)sub_8D2780(v59))
                  && (!(unsigned int)sub_8D2E30(v60) || !(unsigned int)sub_8D2E30(v59))
                  && (!(unsigned int)sub_8D2A90(v60) || !(unsigned int)sub_8D2A90(v59)) )
                {
                  sub_6E68E0(0x550u, (__int64)a1);
                  v19 = a1[1].m128i_i8[0];
                  goto LABEL_33;
                }
LABEL_84:
                if ( v57 )
                {
                  if ( qword_4F077A8 > 0x76BFu && sub_6E53E0(5, 0x550u, &a1[4].m128i_i32[1]) )
                    sub_684B30(0x550u, &a1[4].m128i_i32[1]);
                  v19 = a1[1].m128i_i8[0];
LABEL_33:
                  if ( !v19 )
                    goto LABEL_34;
                  goto LABEL_72;
                }
LABEL_89:
                sub_69D070(0x484u, &a1[4].m128i_i32[1]);
                v19 = a1[1].m128i_i8[0];
                v57 = 0;
                goto LABEL_33;
              }
              *(_QWORD *)&v66[0] = a1[9].m128i_i64[0];
            }
            if ( !v8 )
            {
              v57 = 0;
LABEL_72:
              v23 = a1->m128i_i64[0];
              for ( j = *(_BYTE *)(a1->m128i_i64[0] + 140); j == 12; j = *(_BYTE *)(v23 + 140) )
                v23 = *(_QWORD *)(v23 + 160);
              if ( j )
              {
                v25 = (__int64 *)sub_6EED10(v9, &v61, 0, dword_4F077C0, a2, 0);
                if ( v57 )
                  v25 = (__int64 *)sub_691700((__int64)v25, v57, 0);
                sub_6E7150(v25, (__int64)a1);
                goto LABEL_35;
              }
LABEL_34:
              sub_6E6870((__int64)a1);
LABEL_35:
              sub_6E4BC0((__int64)a1, (__int64)v62);
              result = v63.m128i_i64[0];
              a1[5].m128i_i64[1] = v63.m128i_i64[0];
              return result;
            }
            goto LABEL_22;
          }
          v9 = result;
          goto LABEL_17;
        }
LABEL_12:
        v10 = qword_4F077A8;
        v11 = 119;
        goto LABEL_13;
      }
      v38 = *(__int64 **)(v9 + 72);
    }
    v9 = (__int64)v38;
    goto LABEL_105;
  }
  if ( (_BYTE)result != 1 )
    return result;
LABEL_37:
  result = sub_6ED0A0((__int64)a1);
  if ( !(_DWORD)result && a1[1].m128i_i8[0] == 1 )
  {
    result = (__int64)&dword_4F077BC;
    if ( dword_4F077BC )
    {
      result = (__int64)&qword_4F077A8;
      if ( qword_4F077A8 <= 0x76BFu )
      {
        result = a1[9].m128i_i64[0];
        if ( *(_BYTE *)(result + 24) == 1 && *(_BYTE *)(result + 56) == 6 )
          return sub_69D070(0x550u, &a1[4].m128i_i32[1]);
      }
    }
  }
  return result;
}
