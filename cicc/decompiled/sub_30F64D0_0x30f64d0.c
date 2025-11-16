// Function: sub_30F64D0
// Address: 0x30f64d0
//
__int64 __fastcall sub_30F64D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // rax
  unsigned __int64 v9; // r12
  __int64 v10; // rbx
  __int64 v11; // r13
  __int64 result; // rax
  __int64 v13; // r8
  unsigned __int64 v14; // rcx
  __int64 v15; // rdx
  _QWORD *v16; // rax
  __int64 v17; // rsi
  __int64 v18; // r10
  int v19; // ecx
  bool v20; // sf
  bool v21; // of
  __int64 v22; // rsi
  int v23; // ecx
  __int64 v24; // r14
  __int64 v25; // r11
  __int64 v26; // rax
  const __m128i *v27; // r10
  __int64 v28; // r11
  unsigned __int64 v29; // r8
  const __m128i *v30; // r15
  __int64 v31; // r12
  unsigned __int64 v32; // rax
  int v33; // r8d
  __int64 v34; // rdx
  unsigned __int64 v35; // rcx
  __int64 v36; // rdx
  _QWORD *v37; // rax
  __int64 v38; // rdi
  __int64 v39; // rax
  int v40; // esi
  bool v41; // cc
  __int64 v42; // rdx
  int v43; // edx
  unsigned __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rdx
  unsigned __int64 v47; // rdx
  __int64 v48; // rcx
  unsigned __int64 v49; // rdx
  int v51; // [rsp+18h] [rbp-58h]
  unsigned __int64 v52; // [rsp+20h] [rbp-50h]
  unsigned __int64 v53; // [rsp+28h] [rbp-48h]
  unsigned __int64 v54; // [rsp+28h] [rbp-48h]
  __m128i *v55; // [rsp+30h] [rbp-40h]
  __int64 v56; // [rsp+38h] [rbp-38h]

  v7 = a5;
  v9 = a1;
  v10 = a3;
  if ( a7 <= a5 )
    v7 = a7;
  if ( a4 <= v7 )
  {
LABEL_28:
    result = 0xAAAAAAAAAAAAAAABLL;
    v35 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(a2 - v9) >> 3);
    if ( (__int64)(a2 - v9) > 0 )
    {
      v36 = a6;
      v37 = (_QWORD *)v9;
      do
      {
        v38 = *v37;
        v36 += 24;
        v37 += 3;
        *(_QWORD *)(v36 - 24) = v38;
        *(_QWORD *)(v36 - 16) = *(v37 - 2);
        *(_DWORD *)(v36 - 8) = *((_DWORD *)v37 - 2);
        --v35;
      }
      while ( v35 );
      v39 = 24;
      if ( (__int64)(a2 - v9) > 0 )
        v39 = a2 - v9;
      result = a6 + v39;
      if ( a6 == result || v10 == a2 )
      {
LABEL_41:
        if ( result != a6 )
        {
          result -= a6;
          v44 = 0xAAAAAAAAAAAAAAABLL * (result >> 3);
          if ( result > 0 )
          {
            do
            {
              v45 = *(_QWORD *)a6;
              v9 += 24LL;
              a6 += 24;
              *(_QWORD *)(v9 - 24) = v45;
              *(_QWORD *)(v9 - 16) = *(_QWORD *)(a6 - 16);
              result = *(unsigned int *)(a6 - 8);
              *(_DWORD *)(v9 - 8) = result;
              --v44;
            }
            while ( v44 );
          }
        }
      }
      else
      {
        while ( 1 )
        {
          v40 = *(_DWORD *)(a2 + 16);
          v41 = *(_DWORD *)(a6 + 16) < v40;
          if ( *(_DWORD *)(a6 + 16) == v40 )
            v41 = *(_QWORD *)(a6 + 8) < *(_QWORD *)(a2 + 8);
          if ( v41 )
          {
            v42 = *(_QWORD *)a2;
            a2 += 24;
            *(_QWORD *)v9 = v42;
            *(_QWORD *)(v9 + 8) = *(_QWORD *)(a2 - 16);
            v43 = *(_DWORD *)(a2 - 8);
          }
          else
          {
            v46 = *(_QWORD *)a6;
            a6 += 24;
            *(_QWORD *)v9 = v46;
            *(_QWORD *)(v9 + 8) = *(_QWORD *)(a6 - 16);
            v43 = *(_DWORD *)(a6 - 8);
          }
          *(_DWORD *)(v9 + 16) = v43;
          v9 += 24LL;
          if ( result == a6 )
            break;
          if ( v10 == a2 )
            goto LABEL_41;
        }
      }
    }
  }
  else
  {
    v11 = a5;
    if ( a7 < a5 )
    {
      v24 = a4;
      v25 = a1;
      v56 = a6;
      while ( 1 )
      {
        if ( v11 < v24 )
        {
          v31 = v24 / 2;
          v30 = (const __m128i *)(v25 + 8 * (v24 / 2 + ((v24 + ((unsigned __int64)v24 >> 63)) & 0xFFFFFFFFFFFFFFFELL)));
          v55 = (__m128i *)sub_30F3D20(a2, a3, (__int64)v30);
          v29 = 0xAAAAAAAAAAAAAAABLL * (((char *)v55 - (char *)v27) >> 3);
        }
        else
        {
          v55 = (__m128i *)(a2 + 8 * (v11 / 2 + ((v11 + ((unsigned __int64)v11 >> 63)) & 0xFFFFFFFFFFFFFFFELL)));
          v26 = sub_30F3CB0(v25, a2, (__int64)v55);
          v29 = v11 / 2;
          v30 = (const __m128i *)v26;
          v31 = 0xAAAAAAAAAAAAAAABLL * ((v26 - v28) >> 3);
        }
        v24 -= v31;
        v51 = v28;
        v53 = v29;
        v32 = sub_30F6220(v30, v27, v55, v24, v29, v56, a7);
        v33 = v53;
        v52 = v53;
        v54 = v32;
        sub_30F64D0(v51, (_DWORD)v30, v32, v31, v33, v56, a7);
        v11 -= v52;
        v34 = v11;
        if ( a7 <= v11 )
          v34 = a7;
        if ( v34 >= v24 )
        {
          v10 = a3;
          a6 = v56;
          v9 = v54;
          a2 = (__int64)v55;
          goto LABEL_28;
        }
        if ( a7 >= v11 )
          break;
        a2 = (__int64)v55;
        v25 = v54;
      }
      v10 = a3;
      a6 = v56;
      v9 = v54;
      a2 = (__int64)v55;
    }
    result = 0xAAAAAAAAAAAAAAABLL;
    v13 = v10 - a2;
    v14 = 0xAAAAAAAAAAAAAAABLL * ((v10 - a2) >> 3);
    if ( v10 - a2 > 0 )
    {
      v15 = a6;
      v16 = (_QWORD *)a2;
      do
      {
        v17 = *v16;
        v15 += 24;
        v16 += 3;
        *(_QWORD *)(v15 - 24) = v17;
        *(_QWORD *)(v15 - 16) = *(v16 - 2);
        *(_DWORD *)(v15 - 8) = *((_DWORD *)v16 - 2);
        --v14;
      }
      while ( v14 );
      if ( v13 <= 0 )
        v13 = 24;
      result = a6 + v13;
      if ( v9 == a2 )
      {
        v49 = 0xAAAAAAAAAAAAAAABLL * (v13 >> 3);
        while ( 1 )
        {
          result -= 24;
          *(_QWORD *)(v10 - 24) = v17;
          v10 -= 24;
          *(_QWORD *)(v10 + 8) = *(_QWORD *)(result + 8);
          *(_DWORD *)(v10 + 16) = *(_DWORD *)(result + 16);
          if ( !--v49 )
            break;
          v17 = *(_QWORD *)(result - 24);
        }
      }
      else if ( a6 != result )
      {
        v18 = a2 - 24;
        while ( 1 )
        {
          result -= 24;
          v19 = *(_DWORD *)(result + 16);
          v21 = __OFSUB__(*(_DWORD *)(v18 + 16), v19);
          v20 = *(_DWORD *)(v18 + 16) - v19 < 0;
          if ( *(_DWORD *)(v18 + 16) == v19 )
          {
LABEL_14:
            v22 = *(_QWORD *)(result + 8);
            v21 = __OFSUB__(*(_QWORD *)(v18 + 8), v22);
            v20 = *(_QWORD *)(v18 + 8) - v22 < 0;
          }
          v10 -= 24;
          if ( v20 != v21 )
            break;
LABEL_19:
          *(_QWORD *)v10 = *(_QWORD *)result;
          *(_QWORD *)(v10 + 8) = *(_QWORD *)(result + 8);
          *(_DWORD *)(v10 + 16) = *(_DWORD *)(result + 16);
          if ( a6 == result )
            return result;
        }
        while ( 1 )
        {
          *(_QWORD *)v10 = *(_QWORD *)v18;
          *(_QWORD *)(v10 + 8) = *(_QWORD *)(v18 + 8);
          *(_DWORD *)(v10 + 16) = *(_DWORD *)(v18 + 16);
          if ( v9 == v18 )
            break;
          v18 -= 24;
          v23 = *(_DWORD *)(result + 16);
          if ( *(_DWORD *)(v18 + 16) == v23 )
            goto LABEL_14;
          v10 -= 24;
          if ( *(_DWORD *)(v18 + 16) >= v23 )
            goto LABEL_19;
        }
        result += 24;
        v47 = 0xAAAAAAAAAAAAAAABLL * ((result - a6) >> 3);
        if ( result - a6 > 0 )
        {
          do
          {
            v48 = *(_QWORD *)(result - 24);
            result -= 24;
            v10 -= 24;
            *(_QWORD *)v10 = v48;
            *(_QWORD *)(v10 + 8) = *(_QWORD *)(result + 8);
            *(_DWORD *)(v10 + 16) = *(_DWORD *)(result + 16);
            --v47;
          }
          while ( v47 );
        }
      }
    }
  }
  return result;
}
