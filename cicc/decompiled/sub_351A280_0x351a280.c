// Function: sub_351A280
// Address: 0x351a280
//
__int64 __fastcall sub_351A280(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, int *a6, __int64 a7)
{
  __int64 result; // rax
  __int64 v9; // r12
  __int64 v10; // rbx
  __int64 v11; // r13
  __int64 v12; // rdi
  __int64 v13; // rcx
  int *v14; // rdx
  char *v15; // rax
  int v16; // esi
  char *v17; // r10
  int v18; // edx
  __int64 v19; // r14
  char *v20; // r11
  char *v21; // rax
  char *v22; // r10
  __int64 v23; // r11
  __int64 v24; // r8
  char *v25; // r15
  __int64 v26; // r12
  char *v27; // rax
  int v28; // r8d
  __int64 v29; // rdx
  __int64 v30; // rcx
  int *v31; // rdx
  int v32; // edi
  __int64 v33; // rdx
  int *v34; // rdx
  int v35; // eax
  __int64 v36; // rdx
  int v37; // edx
  __int64 v38; // rdx
  int v39; // ecx
  __int64 v40; // rdx
  int v41; // ecx
  int v43; // [rsp+18h] [rbp-58h]
  __int64 v44; // [rsp+20h] [rbp-50h]
  __int64 v45; // [rsp+28h] [rbp-48h]
  char *v46; // [rsp+28h] [rbp-48h]
  char *v47; // [rsp+30h] [rbp-40h]
  int *v48; // [rsp+38h] [rbp-38h]

  result = a5;
  v9 = (__int64)a1;
  v10 = a3;
  if ( a7 <= a5 )
    result = a7;
  if ( a4 <= result )
  {
LABEL_25:
    v30 = (__int64)&a2[-v9] >> 4;
    if ( (__int64)&a2[-v9] > 0 )
    {
      v31 = a6;
      result = v9;
      do
      {
        v32 = *(_DWORD *)result;
        v31 += 4;
        result += 16;
        *(v31 - 4) = v32;
        *((_QWORD *)v31 - 1) = *(_QWORD *)(result - 8);
        --v30;
      }
      while ( v30 );
      v33 = 16;
      if ( (__int64)&a2[-v9] > 0 )
        v33 = (__int64)&a2[-v9];
      v34 = (int *)((char *)a6 + v33);
      if ( (char *)v10 == a2 || a6 == v34 )
      {
LABEL_38:
        if ( v34 != a6 )
        {
          v36 = (char *)v34 - (char *)a6;
          result = v36 >> 4;
          if ( v36 > 0 )
          {
            do
            {
              v37 = *a6;
              v9 += 16;
              a6 += 4;
              *(_DWORD *)(v9 - 16) = v37;
              *(_QWORD *)(v9 - 8) = *((_QWORD *)a6 - 1);
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
          if ( (unsigned int)*a6 < *(_DWORD *)a2 )
          {
            v35 = *(_DWORD *)a2;
            a2 += 16;
            *(_DWORD *)v9 = v35;
            result = *((_QWORD *)a2 - 1);
          }
          else
          {
            *(_DWORD *)v9 = *a6;
            result = *((_QWORD *)a6 + 1);
            a6 += 4;
          }
          *(_QWORD *)(v9 + 8) = result;
          v9 += 16;
          if ( v34 == a6 )
            break;
          if ( (char *)v10 == a2 )
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
      v19 = a4;
      v20 = a1;
      v48 = a6;
      while ( 1 )
      {
        if ( v19 > v11 )
        {
          v26 = v19 / 2;
          v25 = &v20[16 * (v19 / 2)];
          v47 = (char *)sub_3510F80(a2, a3, v25);
          v24 = (v47 - v22) >> 4;
        }
        else
        {
          v47 = &a2[16 * (v11 / 2)];
          v21 = (char *)sub_3510F30(v20, (__int64)a2, v47);
          v24 = v11 / 2;
          v25 = v21;
          v26 = (__int64)&v21[-v23] >> 4;
        }
        v19 -= v26;
        v43 = v23;
        v45 = v24;
        v27 = sub_351A050(v25, v22, (__int64)v47, v19, v24, v48, a7);
        v28 = v45;
        v44 = v45;
        v46 = v27;
        sub_351A280(v43, (_DWORD)v25, (_DWORD)v27, v26, v28, (_DWORD)v48, a7);
        result = (__int64)v46;
        v11 -= v44;
        v29 = v11;
        if ( a7 <= v11 )
          v29 = a7;
        if ( v29 >= v19 )
        {
          v10 = a3;
          a6 = v48;
          v9 = (__int64)v46;
          a2 = v47;
          goto LABEL_25;
        }
        if ( a7 >= v11 )
          break;
        a2 = v47;
        v20 = v46;
      }
      v10 = a3;
      a6 = v48;
      v9 = (__int64)v46;
      a2 = v47;
    }
    v12 = v10 - (_QWORD)a2;
    v13 = (v10 - (__int64)a2) >> 4;
    if ( v10 - (__int64)a2 > 0 )
    {
      v14 = a6;
      v15 = a2;
      do
      {
        v16 = *(_DWORD *)v15;
        v14 += 4;
        v15 += 16;
        *(v14 - 4) = v16;
        *((_QWORD *)v14 - 1) = *((_QWORD *)v15 - 1);
        --v13;
      }
      while ( v13 );
      if ( v12 <= 0 )
        v12 = 16;
      result = (__int64)a6 + v12;
      if ( (char *)v9 == a2 )
      {
        v40 = v12 >> 4;
        do
        {
          v41 = *(_DWORD *)(result - 16);
          result -= 16;
          v10 -= 16;
          *(_DWORD *)v10 = v41;
          *(_QWORD *)(v10 + 8) = *(_QWORD *)(result + 8);
          --v40;
        }
        while ( v40 );
      }
      else if ( a6 != (int *)result )
      {
        v17 = a2 - 16;
        while ( 1 )
        {
          result -= 16;
          v18 = *(_DWORD *)v17;
          v10 -= 16;
          if ( *(_DWORD *)v17 < *(_DWORD *)result )
            break;
LABEL_16:
          *(_DWORD *)v10 = *(_DWORD *)result;
          *(_QWORD *)(v10 + 8) = *(_QWORD *)(result + 8);
          if ( a6 == (int *)result )
            return result;
        }
        while ( 1 )
        {
          *(_DWORD *)v10 = v18;
          *(_QWORD *)(v10 + 8) = *((_QWORD *)v17 + 1);
          if ( (char *)v9 == v17 )
            break;
          v17 -= 16;
          v10 -= 16;
          v18 = *(_DWORD *)v17;
          if ( *(_DWORD *)v17 >= *(_DWORD *)result )
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
