// Function: sub_148CD40
// Address: 0x148cd40
//
__int64 __fastcall sub_148CD40(_QWORD *a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 v6; // rdx
  int v7; // eax
  __int64 v8; // rcx
  int v9; // edi
  __int64 v10; // rsi
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // r8
  _QWORD *v14; // rax
  _QWORD *v15; // rdx
  __int64 result; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rsi
  __int64 v24; // r8
  __int64 v25; // rax
  __int64 v26; // r14
  __int64 v27; // rsi
  __int64 v28; // r15
  __int64 v29; // rbx
  _QWORD *v30; // rax
  int v31; // eax
  int v32; // r9d
  __int64 v33; // [rsp+8h] [rbp-68h]
  __m128i v34; // [rsp+10h] [rbp-60h] BYREF
  __int64 v35; // [rsp+20h] [rbp-50h]
  __int64 v36; // [rsp+28h] [rbp-48h]
  __int64 v37; // [rsp+30h] [rbp-40h]

  v6 = a1[8];
  v7 = *(_DWORD *)(v6 + 24);
  if ( v7 )
  {
    v8 = *(_QWORD *)(a2 + 40);
    v9 = v7 - 1;
    v10 = *(_QWORD *)(v6 + 8);
    v11 = (v7 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v12 = (__int64 *)(v10 + 16LL * v11);
    v13 = *v12;
    if ( v8 == *v12 )
    {
LABEL_3:
      v14 = (_QWORD *)v12[1];
      if ( v14 )
      {
        v15 = (_QWORD *)a1[1];
        if ( v15 )
        {
          if ( v15 != v14 )
          {
            while ( 1 )
            {
              v14 = (_QWORD *)*v14;
              if ( v15 == v14 )
                break;
              if ( !v14 )
                return sub_145DC80((__int64)a1, a2);
            }
          }
        }
      }
    }
    else
    {
      v31 = 1;
      while ( v13 != -8 )
      {
        v32 = v31 + 1;
        v11 = v9 & (v31 + v11);
        v12 = (__int64 *)(v10 + 16LL * v11);
        v13 = *v12;
        if ( v8 == *v12 )
          goto LABEL_3;
        v31 = v32;
      }
    }
  }
  result = sub_148C2B0((__int64)a1, a2, a3, a4);
  if ( !result )
  {
    result = sub_1482AA0(a1, a2, a3, a4, v17, v18, v19);
    if ( !result )
    {
      v20 = sub_1632FA0(*(_QWORD *)(a1[3] + 40LL));
      v21 = a1[5];
      v22 = a1[7];
      v23 = a1[6];
      v34.m128i_i64[0] = v20;
      v34.m128i_i64[1] = v21;
      v35 = v22;
      v36 = v23;
      v37 = 0;
      v25 = sub_13E3350(a2, &v34, 0, 1, v24);
      v26 = v25;
      if ( v25 )
      {
        if ( *(_BYTE *)(v25 + 16) <= 0x17u )
          return sub_146F1B0((__int64)a1, v26);
        v27 = *(_QWORD *)(v25 + 40);
        v28 = *(_QWORD *)(a2 + 40);
        if ( v27 == v28 )
          return sub_146F1B0((__int64)a1, v26);
        v33 = a1[8];
        v29 = sub_13AE450(v33, v27);
        if ( !v29 )
          return sub_146F1B0((__int64)a1, v26);
        v30 = (_QWORD *)sub_13AE450(v33, v28);
        if ( (_QWORD *)v29 == v30 )
          return sub_146F1B0((__int64)a1, v26);
        while ( v30 )
        {
          v30 = (_QWORD *)*v30;
          if ( (_QWORD *)v29 == v30 )
            return sub_146F1B0((__int64)a1, v26);
        }
      }
      return sub_145DC80((__int64)a1, a2);
    }
  }
  return result;
}
