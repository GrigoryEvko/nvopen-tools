// Function: sub_2B5D270
// Address: 0x2b5d270
//
__int64 __fastcall sub_2B5D270(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6, __int64 a7)
{
  __int64 result; // rax
  __int64 v9; // r12
  __int64 v10; // rbx
  __int64 v11; // r13
  __int64 v12; // r8
  __int64 v13; // rcx
  __int64 *v14; // rdx
  __int64 *v15; // rax
  __int64 v16; // rsi
  __int64 v17; // r10
  __int64 v18; // r14
  __int64 v19; // r11
  __int64 v20; // rax
  char *v21; // r10
  __int64 v22; // r11
  __int64 v23; // r8
  char *v24; // r15
  __int64 v25; // r12
  char *v26; // rax
  int v27; // r8d
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 *v30; // rdx
  __int64 *v31; // rax
  __int64 v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rdx
  int v35; // edx
  __int64 v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // rdx
  int v43; // [rsp+18h] [rbp-58h]
  __int64 v44; // [rsp+20h] [rbp-50h]
  __int64 v45; // [rsp+28h] [rbp-48h]
  char *v46; // [rsp+28h] [rbp-48h]
  __int64 v47; // [rsp+30h] [rbp-40h]
  __int64 *v48; // [rsp+38h] [rbp-38h]

  result = a5;
  v9 = a1;
  v10 = a3;
  if ( a7 <= a5 )
    result = a7;
  if ( a4 <= result )
  {
LABEL_25:
    v29 = (a2 - v9) >> 4;
    if ( a2 - v9 > 0 )
    {
      v30 = a6;
      v31 = (__int64 *)v9;
      do
      {
        v32 = *v31;
        v30 += 2;
        v31 += 2;
        *(v30 - 2) = v32;
        *((_DWORD *)v30 - 2) = *((_DWORD *)v31 - 2);
        --v29;
      }
      while ( v29 );
      v33 = 16;
      if ( a2 - v9 > 0 )
        v33 = a2 - v9;
      result = (__int64)a6 + v33;
      if ( v10 == a2 || a6 == (__int64 *)result )
      {
LABEL_38:
        if ( (__int64 *)result != a6 )
        {
          result -= (__int64)a6;
          v37 = result >> 4;
          if ( result > 0 )
          {
            do
            {
              v38 = *a6;
              v9 += 16;
              a6 += 2;
              *(_QWORD *)(v9 - 16) = v38;
              result = *((unsigned int *)a6 - 2);
              *(_DWORD *)(v9 - 8) = result;
              --v37;
            }
            while ( v37 );
          }
        }
      }
      else
      {
        while ( 1 )
        {
          if ( *(_DWORD *)(a2 + 8) > *((_DWORD *)a6 + 2) )
          {
            v34 = *(_QWORD *)a2;
            a2 += 16;
            *(_QWORD *)v9 = v34;
            v35 = *(_DWORD *)(a2 - 8);
          }
          else
          {
            v36 = *a6;
            a6 += 2;
            *(_QWORD *)v9 = v36;
            v35 = *((_DWORD *)a6 - 2);
          }
          *(_DWORD *)(v9 + 8) = v35;
          v9 += 16;
          if ( (__int64 *)result == a6 )
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
      v18 = a4;
      v19 = a1;
      v48 = a6;
      while ( 1 )
      {
        if ( v18 > v11 )
        {
          v25 = v18 / 2;
          v24 = (char *)(v19 + 16 * (v18 / 2));
          v47 = sub_2B0EE60(a2, a3, (__int64)v24);
          v23 = (v47 - (__int64)v21) >> 4;
        }
        else
        {
          v47 = a2 + 16 * (v11 / 2);
          v20 = sub_2B0EEB0(v19, a2, v47);
          v23 = v11 / 2;
          v24 = (char *)v20;
          v25 = (v20 - v22) >> 4;
        }
        v18 -= v25;
        v43 = v22;
        v45 = v23;
        v26 = sub_2B5D040(v24, v21, v47, v18, v23, v48, a7);
        v27 = v45;
        v44 = v45;
        v46 = v26;
        sub_2B5D270(v43, (_DWORD)v24, (_DWORD)v26, v25, v27, (_DWORD)v48, a7);
        result = (__int64)v46;
        v11 -= v44;
        v28 = v11;
        if ( a7 <= v11 )
          v28 = a7;
        if ( v28 >= v18 )
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
        v19 = (__int64)v46;
      }
      v10 = a3;
      a6 = v48;
      v9 = (__int64)v46;
      a2 = v47;
    }
    v12 = v10 - a2;
    v13 = (v10 - a2) >> 4;
    if ( v10 - a2 > 0 )
    {
      v14 = a6;
      v15 = (__int64 *)a2;
      do
      {
        v16 = *v15;
        v14 += 2;
        v15 += 2;
        *(v14 - 2) = v16;
        *((_DWORD *)v14 - 2) = *((_DWORD *)v15 - 2);
        --v13;
      }
      while ( v13 );
      if ( v12 <= 0 )
        v12 = 16;
      result = (__int64)a6 + v12;
      if ( v9 == a2 )
      {
        v41 = v12 >> 4;
        while ( 1 )
        {
          result -= 16;
          *(_QWORD *)(v10 - 16) = v16;
          v10 -= 16;
          *(_DWORD *)(v10 + 8) = *(_DWORD *)(result + 8);
          if ( !--v41 )
            break;
          v16 = *(_QWORD *)(result - 16);
        }
      }
      else if ( a6 != (__int64 *)result )
      {
        v17 = a2 - 16;
        while ( 1 )
        {
          result -= 16;
          v10 -= 16;
          if ( *(_DWORD *)(result + 8) > *(_DWORD *)(v17 + 8) )
            break;
LABEL_16:
          *(_QWORD *)v10 = *(_QWORD *)result;
          *(_DWORD *)(v10 + 8) = *(_DWORD *)(result + 8);
          if ( a6 == (__int64 *)result )
            return result;
        }
        while ( 1 )
        {
          *(_QWORD *)v10 = *(_QWORD *)v17;
          *(_DWORD *)(v10 + 8) = *(_DWORD *)(v17 + 8);
          if ( v9 == v17 )
            break;
          v17 -= 16;
          v10 -= 16;
          if ( *(_DWORD *)(result + 8) <= *(_DWORD *)(v17 + 8) )
            goto LABEL_16;
        }
        result += 16;
        v39 = (result - (__int64)a6) >> 4;
        if ( result - (__int64)a6 > 0 )
        {
          do
          {
            v40 = *(_QWORD *)(result - 16);
            result -= 16;
            v10 -= 16;
            *(_QWORD *)v10 = v40;
            *(_DWORD *)(v10 + 8) = *(_DWORD *)(result + 8);
            --v39;
          }
          while ( v39 );
        }
      }
    }
  }
  return result;
}
