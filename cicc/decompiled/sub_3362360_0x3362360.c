// Function: sub_3362360
// Address: 0x3362360
//
__int64 __fastcall sub_3362360(char *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int *a6, __int64 a7)
{
  __int64 result; // rax
  char *v9; // r12
  __int64 v10; // rbx
  __int64 v11; // r13
  __int64 v12; // r8
  __int64 v13; // rcx
  int *v14; // rdx
  int *v15; // rax
  int v16; // esi
  __int64 v17; // r10
  unsigned int v18; // ecx
  unsigned int v19; // edx
  __int64 v20; // r14
  char *v21; // r11
  char *v22; // rax
  char *v23; // r10
  __int64 v24; // r11
  __int64 v25; // r8
  char *v26; // r15
  __int64 v27; // r12
  char *v28; // rax
  int v29; // r8d
  __int64 v30; // rdx
  __int64 v31; // rcx
  int *v32; // rdx
  int v33; // edi
  __int64 v34; // rcx
  int *v35; // rcx
  __int64 v36; // rcx
  int v37; // edx
  __int64 v38; // rdx
  int v39; // ecx
  __int64 v40; // rdx
  int v42; // [rsp+18h] [rbp-58h]
  __int64 v43; // [rsp+20h] [rbp-50h]
  __int64 v44; // [rsp+28h] [rbp-48h]
  char *v45; // [rsp+28h] [rbp-48h]
  char *v46; // [rsp+30h] [rbp-40h]
  int *v47; // [rsp+38h] [rbp-38h]

  result = a5;
  v9 = a1;
  v10 = a3;
  if ( a7 <= a5 )
    result = a7;
  if ( result >= a4 )
  {
LABEL_25:
    v31 = (a2 - (__int64)v9) >> 4;
    if ( a2 - (__int64)v9 > 0 )
    {
      v32 = a6;
      result = (__int64)v9;
      do
      {
        v33 = *(_DWORD *)result;
        v32 += 4;
        result += 16;
        *(v32 - 4) = v33;
        *((_QWORD *)v32 - 1) = *(_QWORD *)(result - 8);
        --v31;
      }
      while ( v31 );
      v34 = 16;
      if ( a2 - (__int64)v9 > 0 )
        v34 = a2 - (_QWORD)v9;
      v35 = (int *)((char *)a6 + v34);
      if ( v10 == a2 || a6 == v35 )
      {
LABEL_38:
        if ( v35 != a6 )
        {
          v36 = (char *)v35 - (char *)a6;
          result = v36 >> 4;
          if ( v36 > 0 )
          {
            do
            {
              v37 = *a6;
              v9 += 16;
              a6 += 4;
              *((_DWORD *)v9 - 4) = v37;
              *((_QWORD *)v9 - 1) = *((_QWORD *)a6 - 1);
              --result;
            }
            while ( result );
          }
        }
      }
      else
      {
        while ( 1 )
        {
          if ( *(_DWORD *)a2 < (unsigned int)*a6 )
          {
            *(_DWORD *)v9 = *(_DWORD *)a2;
            result = *(_QWORD *)(a2 + 8);
            a2 += 16;
          }
          else
          {
            *(_DWORD *)v9 = *a6;
            result = *((_QWORD *)a6 + 1);
            a6 += 4;
          }
          *((_QWORD *)v9 + 1) = result;
          v9 += 16;
          if ( v35 == a6 )
            break;
          if ( v10 == a2 )
            goto LABEL_38;
        }
      }
    }
  }
  else
  {
    v11 = a5;
    if ( a7 < a5 )
    {
      v20 = a4;
      v21 = a1;
      v47 = a6;
      while ( 1 )
      {
        if ( v20 > v11 )
        {
          v27 = v20 / 2;
          v26 = &v21[16 * (v20 / 2)];
          v46 = (char *)sub_335CB40((_DWORD *)a2, a3, v26);
          v25 = (v46 - v23) >> 4;
        }
        else
        {
          v46 = (char *)(a2 + 16 * (v11 / 2));
          v22 = (char *)sub_335CAF0(v21, a2, v46);
          v25 = v11 / 2;
          v26 = v22;
          v27 = (__int64)&v22[-v24] >> 4;
        }
        v20 -= v27;
        v42 = v24;
        v44 = v25;
        v28 = sub_3362130(v26, v23, (__int64)v46, v20, v25, v47, a7);
        v29 = v44;
        v43 = v44;
        v45 = v28;
        sub_3362360(v42, (_DWORD)v26, (_DWORD)v28, v27, v29, (_DWORD)v47, a7);
        result = (__int64)v45;
        v11 -= v43;
        v30 = v11;
        if ( a7 <= v11 )
          v30 = a7;
        if ( v30 >= v20 )
        {
          v10 = a3;
          a6 = v47;
          v9 = v45;
          a2 = (__int64)v46;
          goto LABEL_25;
        }
        if ( a7 >= v11 )
          break;
        a2 = (__int64)v46;
        v21 = v45;
      }
      v10 = a3;
      a6 = v47;
      v9 = v45;
      a2 = (__int64)v46;
    }
    v12 = v10 - a2;
    v13 = (v10 - a2) >> 4;
    if ( v10 - a2 > 0 )
    {
      v14 = a6;
      v15 = (int *)a2;
      do
      {
        v16 = *v15;
        v14 += 4;
        v15 += 4;
        *(v14 - 4) = v16;
        *((_QWORD *)v14 - 1) = *((_QWORD *)v15 - 1);
        --v13;
      }
      while ( v13 );
      if ( v12 <= 0 )
        v12 = 16;
      result = (__int64)a6 + v12;
      if ( v9 == (char *)a2 )
      {
        v40 = v12 >> 4;
        while ( 1 )
        {
          result -= 16;
          *(_DWORD *)(v10 - 16) = v16;
          v10 -= 16;
          *(_QWORD *)(v10 + 8) = *(_QWORD *)(result + 8);
          if ( !--v40 )
            break;
          v16 = *(_DWORD *)(result - 16);
        }
      }
      else if ( a6 != (int *)result )
      {
        v17 = a2 - 16;
        while ( 1 )
        {
          v18 = *(_DWORD *)v17;
          v19 = *(_DWORD *)(result - 16);
          result -= 16;
          v10 -= 16;
          if ( v19 < *(_DWORD *)v17 )
            break;
LABEL_16:
          *(_DWORD *)v10 = v19;
          *(_QWORD *)(v10 + 8) = *(_QWORD *)(result + 8);
          if ( a6 == (int *)result )
            return result;
        }
        while ( 1 )
        {
          *(_DWORD *)v10 = v18;
          *(_QWORD *)(v10 + 8) = *(_QWORD *)(v17 + 8);
          if ( v9 == (char *)v17 )
            break;
          v19 = *(_DWORD *)result;
          v18 = *(_DWORD *)(v17 - 16);
          v17 -= 16;
          v10 -= 16;
          if ( *(_DWORD *)result >= v18 )
            goto LABEL_16;
        }
        result += 16;
        v38 = (result - (__int64)a6) >> 4;
        if ( result - (__int64)a6 > 0 )
        {
          do
          {
            v39 = *(_DWORD *)(result - 16);
            result -= 16;
            v10 -= 16;
            *(_DWORD *)v10 = v39;
            *(_QWORD *)(v10 + 8) = *(_QWORD *)(result + 8);
            --v38;
          }
          while ( v38 );
        }
      }
    }
  }
  return result;
}
