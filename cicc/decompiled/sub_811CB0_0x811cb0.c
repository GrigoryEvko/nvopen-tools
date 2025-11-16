// Function: sub_811CB0
// Address: 0x811cb0
//
_QWORD *__fastcall sub_811CB0(_QWORD *a1, __int64 a2, int a3, _QWORD *a4)
{
  _QWORD *v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r12
  _QWORD *v10; // rax
  _QWORD *result; // rax
  char v12; // dl
  __int64 v13; // r14
  __m128i *v14; // r8
  __int8 v15; // al
  int v16; // eax
  _QWORD *v17; // rdi
  __int64 v18; // rax
  __int64 v19; // r14
  _QWORD *v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // r14
  __int64 v24; // rax
  int v25; // eax
  __m128i *v26; // [rsp+8h] [rbp-68h]
  __m128i *v27; // [rsp+8h] [rbp-68h]
  __m128i *v28; // [rsp+8h] [rbp-68h]
  int v30; // [rsp+2Ch] [rbp-44h] BYREF
  _QWORD *v31; // [rsp+30h] [rbp-40h] BYREF
  __int64 *v32[7]; // [rsp+38h] [rbp-38h] BYREF

  if ( a3 && (!dword_4D0425C || qword_4F077A8 > 0xC34Fu || (_DWORD)qword_4F077B4) )
  {
    v6 = (_QWORD *)qword_4F18BE0;
    ++*a4;
    v7 = v6[2];
    if ( (unsigned __int64)(v7 + 1) > v6[1] )
    {
      sub_823810(v6);
      v6 = (_QWORD *)qword_4F18BE0;
      v7 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v6[4] + v7) = 74;
    ++v6[2];
  }
  else
  {
    v6 = (_QWORD *)qword_4F18BE0;
    ++*a4;
    v8 = v6[2];
    if ( (unsigned __int64)(v8 + 1) > v6[1] )
    {
      sub_823810(v6);
      v6 = (_QWORD *)qword_4F18BE0;
      v8 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v6[4] + v8) = 73;
    ++v6[2];
  }
  v9 = 0;
  v10 = (_QWORD *)*a1;
  v31 = v10;
  if ( v10 )
  {
    while ( 1 )
    {
      if ( a2 && ((*(_BYTE *)(a2 + 33) & 2) == 0 || *(_QWORD *)(a2 + 24) <= v9) || a3 && (v10[3] & 8) == 0 )
      {
LABEL_14:
        v6 = (_QWORD *)qword_4F18BE0;
        break;
      }
      v12 = *((_BYTE *)v10 + 8);
      switch ( v12 )
      {
        case 0:
          v13 = v10[4];
          if ( v13 )
          {
            if ( v10[2] || (v10[3] & 0x10) != 0 )
            {
              if ( !(unsigned int)sub_80C5A0(v10[4], 6, 1, 0, v32, a4) )
              {
                *a4 += 2LL;
                sub_8238B0(qword_4F18BE0, &unk_3C1BB44, 2);
                sub_80F5E0(v13, 0, a4);
                if ( !a4[5] )
                  sub_80A250(v13, 6, 1, (__int64)a4);
              }
            }
            else
            {
              sub_80F5E0(v10[4], 0, a4);
            }
          }
          break;
        case 2:
          v23 = sub_89A800(v10[4]);
          v24 = *(_QWORD *)(v23 + 168);
          if ( v24 && (*(_BYTE *)(v24 + 160) & 4) != 0 )
            break;
          if ( *(_BYTE *)(v23 + 120) == 8 )
          {
            if ( (unsigned int)sub_80C5A0(v23, 59, 0, 0, v32, a4) )
              break;
            sub_812B60(v23 + 128, 0, a4);
            if ( a4[5] )
              break;
          }
          else
          {
            if ( (unsigned int)sub_80C5A0(v23, 59, 0, 0, v32, a4) )
              break;
            v30 = 0;
            sub_811730(v23, 0x3Bu, &v30, (__int64 *)v32, 0, (__int64)a4);
            sub_80BC40(*(char **)(v23 + 8), a4);
            sub_80C110(v30, v32[0], a4);
            if ( a4[5] )
              break;
          }
          sub_80A250(v23, 59, 0, (__int64)a4);
          break;
        case 3:
          v31 = (_QWORD *)*v10;
          sub_811CB0(&v31, 0, 1, a4);
          v10 = v31;
          goto LABEL_28;
        case 1:
          v14 = (__m128i *)v10[4];
          v15 = v14[10].m128i_i8[13];
          if ( v15 == 12 )
          {
            v28 = v14;
            v25 = sub_72E9D0(v14, v32, &v30);
            v14 = v28;
            if ( v25 && !v30 )
              v14 = (__m128i *)v32[0];
            v15 = v14[10].m128i_i8[13];
          }
          if ( ((v15 - 10) & 0xFD) == 0
            || v15 == 7
            || v15 == 6
            && ((v26 = v14, v16 = sub_8D2FB0(v14[8].m128i_i64[0]), v14 = v26, !v16)
             || dword_4D0425C && unk_4D04250 <= 0x76BFu) )
          {
            v17 = (_QWORD *)qword_4F18BE0;
            ++*a4;
            v18 = v17[2];
            if ( (unsigned __int64)(v18 + 1) > v17[1] )
            {
              v27 = v14;
              sub_823810(v17);
              v17 = (_QWORD *)qword_4F18BE0;
              v14 = v27;
              v18 = *(_QWORD *)(qword_4F18BE0 + 16);
            }
            *(_BYTE *)(v17[4] + v18) = 88;
            v19 = v17[2];
            v17[2] = v19 + 1;
            sub_80D8A0(v14, 0, 0, a4);
            v20 = (_QWORD *)qword_4F18BE0;
            v21 = *(_QWORD *)(qword_4F18BE0 + 32);
            if ( *(_BYTE *)(v21 + v19 + 1) != 76
              || dword_4D0425C && unk_4D04250 <= 0x76BFu
              || *(_BYTE *)(v21 + v19 + 2) == 95 && *(_BYTE *)(v21 + v19 + 3) == 90 && (_DWORD)qword_4F077B4 )
            {
              ++*a4;
              v22 = v20[2];
              if ( (unsigned __int64)(v22 + 1) > v20[1] )
              {
                sub_823810(v20);
                v20 = (_QWORD *)qword_4F18BE0;
                v22 = *(_QWORD *)(qword_4F18BE0 + 16);
              }
              *(_BYTE *)(v20[4] + v22) = 69;
              ++v20[2];
            }
            else
            {
              *(_BYTE *)(v21 + v19) = 32;
              ++a4[1];
              --*a4;
            }
          }
          else
          {
            sub_80D8A0(v14, 0, 0, a4);
          }
          break;
        default:
          sub_721090();
      }
      v10 = (_QWORD *)*v31;
      v31 = (_QWORD *)*v31;
LABEL_28:
      ++v9;
      if ( !v10 )
        goto LABEL_14;
    }
  }
  ++*a4;
  result = (_QWORD *)v6[2];
  if ( (unsigned __int64)result + 1 > v6[1] )
  {
    sub_823810(v6);
    v6 = (_QWORD *)qword_4F18BE0;
    result = *(_QWORD **)(qword_4F18BE0 + 16);
  }
  *((_BYTE *)result + v6[4]) = 69;
  ++v6[2];
  if ( a3 )
  {
    result = v31;
    *a1 = v31;
  }
  return result;
}
