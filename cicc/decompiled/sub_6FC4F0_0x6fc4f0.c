// Function: sub_6FC4F0
// Address: 0x6fc4f0
//
__int64 __fastcall sub_6FC4F0(__m128i *a1, __m128i *a2, _DWORD *a3, __int64 *a4)
{
  __int64 v6; // r15
  __int64 v7; // r14
  _BOOL4 v8; // eax
  __int64 result; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rax
  __m128i *v17; // rdi
  _BOOL4 v18; // eax
  _BOOL4 v19; // [rsp+4h] [rbp-5Ch]
  _BOOL4 v20; // [rsp+4h] [rbp-5Ch]
  _QWORD v22[10]; // [rsp+10h] [rbp-50h] BYREF

  v6 = a2->m128i_i64[0];
  v7 = a1->m128i_i64[0];
  if ( !qword_4D0495C )
  {
    if ( (unsigned int)sub_8D2660(v7) || a1[1].m128i_i8[0] == 2 && (unsigned int)sub_712570(&a1[9]) )
    {
      v17 = (__m128i *)a2->m128i_i64[0];
      if ( (unsigned int)sub_8D2660(a2->m128i_i64[0])
        || a2[1].m128i_i8[0] == 2 && (v17 = a2 + 9, (unsigned int)sub_712570(&a2[9])) )
      {
        sub_721090(v17);
      }
      goto LABEL_30;
    }
    if ( (unsigned int)sub_8D2660(a2->m128i_i64[0]) || a2[1].m128i_i8[0] == 2 && (unsigned int)sub_712570(&a2[9]) )
      goto LABEL_4;
    v16 = sub_8E1ED0(v7, v6, v12, v13, v14, v15);
    *a4 = v16;
    if ( dword_4D04964 | (v16 != 0) )
    {
      if ( v16 )
        goto LABEL_5;
      goto LABEL_18;
    }
  }
  if ( !(unsigned int)sub_8D3D10(v7)
    || (v19 = a2[1].m128i_i8[0] == 2,
        v8 = sub_6EB660((__int64)a2),
        !(unsigned int)sub_8E0CC0(v6, v19, v8, (int)a2 + 144, v7, 1, (__int64)v22)) )
  {
    if ( (unsigned int)sub_8D3D10(v6) )
    {
      v20 = a1[1].m128i_i8[0] == 2;
      v18 = sub_6EB660((__int64)a1);
      if ( (unsigned int)sub_8E0CC0(v7, v20, v18, (int)a1 + 144, v6, 1, (__int64)v22) )
      {
LABEL_30:
        *a4 = v6;
        goto LABEL_5;
      }
    }
LABEL_18:
    sub_6E5ED0(0x2Au, a3, v7, v6);
    *a4 = sub_72C930(42);
    return 0;
  }
LABEL_4:
  *a4 = v7;
LABEL_5:
  result = 1;
  if ( qword_4D0495C
    && v22[0]
    && (*(_BYTE *)(v22[0] + 96LL) & 4) == 0
    && ((*(_BYTE *)(v22[0] + 96LL) & 2) != 0
     || (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v22[0] + 112LL) + 8LL) + 16LL) + 96LL) & 2) != 0) )
  {
    v10 = *a4;
    if ( *a4 == v7 || v7 && v10 && dword_4F07588 && (v11 = *(_QWORD *)(v10 + 32), *(_QWORD *)(v7 + 32) == v11) && v11 )
    {
      sub_6FC3F0(v6, a1, 0);
      *a4 = v6;
      return 1;
    }
    else
    {
      sub_6FC3F0(v7, a2, 0);
      *a4 = v7;
      return 1;
    }
  }
  return result;
}
