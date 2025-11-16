// Function: sub_8441D0
// Address: 0x8441d0
//
__int64 __fastcall sub_8441D0(__m128i *a1, __int64 a2, int a3, _BYTE *a4, __int64 *a5, _DWORD *a6)
{
  __m128i *v9; // r15
  __int64 v10; // rdx
  int v11; // r8d
  __int64 i; // rax
  __int64 *v13; // r15
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rax
  __int64 v19; // r13
  __int64 result; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  int v28; // eax
  int v29; // eax
  int v31; // [rsp+18h] [rbp-38h]

  v9 = *(__m128i **)(*(_QWORD *)(a2 + 40) + 32LL);
  LODWORD(v10) = (_DWORD)v9;
  if ( a3 )
    v10 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 216) + 40LL) + 32LL);
  *a6 = 0;
  v11 = (_DWORD)a1 + 68;
  if ( (*(_BYTE *)(a2 + 194) & 4) == 0 || (*(_BYTE *)(a2 + 206) & 0x10) != 0 || (*(_BYTE *)(a2 + 193) & 4) != 0 )
    goto LABEL_5;
  if ( !a4 || (a4[16] & 0x88) != 0 )
  {
    v31 = v10;
    v28 = sub_8D3A70(a1->m128i_i64[0]);
    LODWORD(v10) = v31;
    v11 = (_DWORD)a1 + 68;
    if ( v28 )
    {
      v29 = sub_8D5DF0(a1->m128i_i64[0]);
      LODWORD(v10) = v31;
      v11 = (_DWORD)a1 + 68;
      if ( v29 )
      {
        *a6 = 1;
        sub_6E6130(*(_QWORD *)a2, (_DWORD)a1 + 68, v31, 0);
        if ( !a4 )
        {
LABEL_22:
          sub_8424A0(a1, v9, v21, v22, v23);
          goto LABEL_16;
        }
        goto LABEL_13;
      }
    }
LABEL_5:
    for ( i = *(_QWORD *)(a2 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v13 = **(__int64 ***)(i + 168);
    sub_6E6130(*(_QWORD *)a2, v11, v10, 0);
    if ( v13 )
    {
      sub_843D70(a1, (__int64)v13, a4, 0xA7u);
      v18 = sub_6F6F40(a1, 0, v14, v15, v16, v17);
      *a5 = v18;
      v19 = v18;
      result = sub_6E1DA0(a2, *v13);
      *(_QWORD *)(v19 + 16) = result;
      return result;
    }
    sub_6FE880(a1, 1);
    goto LABEL_16;
  }
  *a6 = 1;
  sub_6E6130(*(_QWORD *)a2, (_DWORD)a1 + 68, v10, 0);
LABEL_13:
  if ( !*(_QWORD *)a4 && (*((_WORD *)a4 + 8) & 0x101) == 0 )
    goto LABEL_22;
  a4[16] &= ~4u;
  sub_8449E0(a1, v9, a4, 0, 0);
LABEL_16:
  result = sub_6F6F40(a1, 0, v24, v25, v26, v27);
  *a5 = result;
  return result;
}
