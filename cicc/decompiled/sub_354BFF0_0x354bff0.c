// Function: sub_354BFF0
// Address: 0x354bff0
//
__int64 __fastcall sub_354BFF0(__int64 a1, __int64 a2, int *a3, int *a4, int a5, __int64 a6)
{
  __int64 result; // rax
  int *v10; // rax
  _QWORD *v11; // r12
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // r12
  const __m128i *v15; // rbx
  int v16; // eax
  __int64 v17; // rax
  int *v18; // r12
  const __m128i *v19; // r14
  __int64 v20; // rbx
  __int32 v21; // eax
  __int64 *v22; // rax
  __int64 *v23; // rsi
  unsigned __int64 v24; // rdx
  _QWORD *v25; // rax
  _QWORD *v26; // rdi
  _QWORD *v27; // rcx
  _QWORD *v28; // r8
  int v29; // ecx
  _QWORD *v30; // rcx
  _QWORD *v31; // r8
  int v32; // eax
  int v33; // eax
  int v34; // [rsp+14h] [rbp-8Ch]
  _QWORD **v35; // [rsp+18h] [rbp-88h]
  _QWORD *v37; // [rsp+28h] [rbp-78h]
  __int64 v38; // [rsp+30h] [rbp-70h]
  __int64 v39; // [rsp+38h] [rbp-68h]
  __int64 v40; // [rsp+38h] [rbp-68h]
  _QWORD *v41; // [rsp+40h] [rbp-60h]
  _QWORD *v42; // [rsp+40h] [rbp-60h]
  _QWORD *v44; // [rsp+50h] [rbp-50h]
  int v46; // [rsp+5Ch] [rbp-44h]
  int v47[13]; // [rsp+6Ch] [rbp-34h] BYREF

  v44 = *(_QWORD **)(a6 + 3464);
  result = *(unsigned int *)(a1 + 80);
  v46 = result;
  v34 = a5 - 1;
  if ( (int)result <= *(_DWORD *)(a1 + 84) )
  {
    while ( 1 )
    {
      v47[0] = v46;
      v10 = sub_354BE50(a1, v47);
      v11 = (_QWORD *)*((_QWORD *)v10 + 2);
      v38 = *((_QWORD *)v10 + 4);
      v35 = (_QWORD **)*((_QWORD *)v10 + 5);
      v37 = (_QWORD *)*((_QWORD *)v10 + 6);
LABEL_3:
      if ( v37 != v11 )
        break;
LABEL_41:
      result = (unsigned int)++v46;
      if ( *(_DWORD *)(a1 + 84) < v46 )
        return result;
    }
    while ( 1 )
    {
      v12 = *v11;
      v13 = sub_35459D0(v44, a2);
      if ( *(_QWORD *)v13 != *(_QWORD *)v13 + 32LL * *(unsigned int *)(v13 + 8) )
      {
        v41 = v11;
        v14 = *(_QWORD *)v13 + 32LL * *(unsigned int *)(v13 + 8);
        v39 = a2;
        v15 = *(const __m128i **)v13;
        do
        {
          while ( v12 != (v15->m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) )
          {
            v15 += 2;
            if ( (const __m128i *)v14 == v15 )
              goto LABEL_12;
          }
          if ( (unsigned __int8)sub_3544720(a6, (__int64)v15) )
          {
            v33 = v34 + sub_3545A70(a1, v15, v44);
            if ( v33 > *a4 )
              v33 = *a4;
            *a4 = v33;
          }
          v16 = v15[1].m128i_i32[1] + v46 - v15[1].m128i_i32[2] * a5;
          if ( v16 < *a3 )
            v16 = *a3;
          v15 += 2;
          *a3 = v16;
        }
        while ( (const __m128i *)v14 != v15 );
LABEL_12:
        v11 = v41;
        a2 = v39;
      }
      v17 = sub_3545E90(v44, a2);
      if ( *(_QWORD *)v17 + 32LL * *(unsigned int *)(v17 + 8) != *(_QWORD *)v17 )
      {
        v42 = v11;
        v18 = a4;
        v19 = *(const __m128i **)v17;
        v40 = a2;
        v20 = *(_QWORD *)v17 + 32LL * *(unsigned int *)(v17 + 8);
        do
        {
          while ( v12 != v19->m128i_i64[0] )
          {
            v19 += 2;
            if ( (const __m128i *)v20 == v19 )
              goto LABEL_21;
          }
          if ( (unsigned __int8)sub_3544720(a6, (__int64)v19) )
          {
            v32 = sub_3546200(a1, v19, v44) + 1 - a5;
            if ( v32 < *a3 )
              v32 = *a3;
            *a3 = v32;
          }
          v21 = v46 + v19[1].m128i_i32[2] * a5 - v19[1].m128i_i32[1];
          if ( v21 > *v18 )
            v21 = *v18;
          v19 += 2;
          *v18 = v21;
        }
        while ( (const __m128i *)v20 != v19 );
LABEL_21:
        a4 = v18;
        a2 = v40;
        v11 = v42;
      }
      v22 = *(__int64 **)(v12 + 40);
      v23 = &v22[2 * *(unsigned int *)(v12 + 48)];
      if ( v22 == v23 )
      {
LABEL_25:
        v24 = 0;
      }
      else
      {
        while ( 1 )
        {
          if ( ((*v22 >> 1) & 3) == 1 )
          {
            v24 = *v22 & 0xFFFFFFFFFFFFFFF8LL;
            if ( *(_WORD *)(*(_QWORD *)v24 + 68LL) == 68 || !*(_WORD *)(*(_QWORD *)v24 + 68LL) )
            {
              v30 = *(_QWORD **)(v24 + 120);
              v31 = &v30[2 * *(unsigned int *)(v24 + 128)];
              if ( v30 != v31 )
                break;
            }
          }
LABEL_24:
          v22 += 2;
          if ( v23 == v22 )
            goto LABEL_25;
        }
        while ( (*v30 & 6) != 0
             || *(_WORD *)(*(_QWORD *)(*v30 & 0xFFFFFFFFFFFFFFF8LL) + 68LL)
             && *(_WORD *)(*(_QWORD *)(*v30 & 0xFFFFFFFFFFFFFFF8LL) + 68LL) != 68 )
        {
          v30 += 2;
          if ( v31 == v30 )
            goto LABEL_24;
        }
      }
      v25 = *(_QWORD **)(a2 + 40);
      v26 = &v25[2 * *(unsigned int *)(a2 + 48)];
      if ( v26 != v25 )
        break;
LABEL_39:
      if ( (_QWORD *)v38 != ++v11 )
        goto LABEL_3;
      v11 = *++v35;
      v38 = (__int64)(*v35 + 64);
      if ( v37 == *v35 )
        goto LABEL_41;
    }
    while ( 1 )
    {
      while ( !v24
           || v24 != (*v25 & 0xFFFFFFFFFFFFFFF8LL)
           || !*(_WORD *)(*(_QWORD *)a2 + 68LL)
           || *(_WORD *)(*(_QWORD *)a2 + 68LL) == 68 )
      {
LABEL_28:
        v25 += 2;
        if ( v25 == v26 )
          goto LABEL_39;
      }
      v27 = *(_QWORD **)(a2 + 40);
      v28 = &v27[2 * *(unsigned int *)(a2 + 48)];
      if ( v27 != v28 )
      {
        while ( v12 != (*v27 & 0xFFFFFFFFFFFFFFF8LL) )
        {
          v27 += 2;
          if ( v28 == v27 )
            goto LABEL_36;
        }
        goto LABEL_28;
      }
LABEL_36:
      v29 = v46;
      if ( *a4 <= v46 )
        v29 = *a4;
      v25 += 2;
      *a4 = v29;
      if ( v25 == v26 )
        goto LABEL_39;
    }
  }
  return result;
}
