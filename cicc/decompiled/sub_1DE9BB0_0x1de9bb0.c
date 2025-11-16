// Function: sub_1DE9BB0
// Address: 0x1de9bb0
//
__int64 __fastcall sub_1DE9BB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 result; // rax
  __int64 v9; // r12
  __int64 v10; // rbx
  __int64 v11; // r13
  __int64 v12; // rdi
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rax
  int v16; // esi
  __int64 v17; // r10
  unsigned int v18; // edx
  __int64 v19; // r14
  __int64 v20; // r11
  __int64 v21; // rax
  __int64 v22; // r10
  __int64 v23; // r11
  __int64 v24; // r8
  __int64 *v25; // r15
  __int64 v26; // r12
  __int64 *v27; // rax
  int v28; // r8d
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // rdx
  int v32; // edi
  __int64 v33; // rdx
  __int64 v34; // rdx
  int v35; // eax
  unsigned int v36; // eax
  __int64 v37; // rdx
  int v38; // edx
  __int64 v39; // rdx
  int v40; // ecx
  __int64 v41; // rdx
  int v42; // ecx
  int v44; // [rsp+18h] [rbp-58h]
  __int64 v45; // [rsp+20h] [rbp-50h]
  __int64 v46; // [rsp+28h] [rbp-48h]
  __int64 *v47; // [rsp+28h] [rbp-48h]
  __int64 v48; // [rsp+30h] [rbp-40h]
  __int64 v49; // [rsp+38h] [rbp-38h]

  result = a5;
  v9 = a1;
  v10 = a3;
  if ( a7 <= a5 )
    result = a7;
  if ( a4 <= result )
  {
LABEL_25:
    v30 = (a2 - v9) >> 4;
    if ( a2 - v9 > 0 )
    {
      v31 = a6;
      result = v9;
      do
      {
        v32 = *(_DWORD *)(result + 8);
        v31 += 16;
        result += 16;
        *(_DWORD *)(v31 - 8) = v32;
        *(_QWORD *)(v31 - 16) = *(_QWORD *)(result - 16);
        --v30;
      }
      while ( v30 );
      v33 = 16;
      if ( a2 - v9 > 0 )
        v33 = a2 - v9;
      v34 = a6 + v33;
      if ( v10 == a2 || a6 == v34 )
      {
LABEL_38:
        if ( v34 != a6 )
        {
          v37 = v34 - a6;
          result = v37 >> 4;
          if ( v37 > 0 )
          {
            do
            {
              v38 = *(_DWORD *)(a6 + 8);
              v9 += 16;
              a6 += 16;
              *(_DWORD *)(v9 - 8) = v38;
              *(_QWORD *)(v9 - 16) = *(_QWORD *)(a6 - 16);
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
          v36 = *(_DWORD *)(a6 + 8);
          if ( v36 < *(_DWORD *)(a2 + 8) )
          {
            v35 = *(_DWORD *)(a2 + 8);
            a2 += 16;
            *(_DWORD *)(v9 + 8) = v35;
            result = *(_QWORD *)(a2 - 16);
          }
          else
          {
            *(_DWORD *)(v9 + 8) = v36;
            result = *(_QWORD *)a6;
            a6 += 16;
          }
          *(_QWORD *)v9 = result;
          v9 += 16;
          if ( v34 == a6 )
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
      v19 = a4;
      v20 = a1;
      v49 = a6;
      while ( 1 )
      {
        if ( v19 > v11 )
        {
          v26 = v19 / 2;
          v25 = (__int64 *)(v20 + 16 * (v19 / 2));
          v48 = sub_1DE41C0(a2, a3, (__int64)v25);
          v24 = (v48 - v22) >> 4;
        }
        else
        {
          v48 = a2 + 16 * (v11 / 2);
          v21 = sub_1DE4210(v20, a2, v48);
          v24 = v11 / 2;
          v25 = (__int64 *)v21;
          v26 = (v21 - v23) >> 4;
        }
        v19 -= v26;
        v44 = v23;
        v46 = v24;
        v27 = sub_1DE9970(v25, v22, v48, v19, v24, v49, a7);
        v28 = v46;
        v45 = v46;
        v47 = v27;
        sub_1DE9BB0(v44, (_DWORD)v25, (_DWORD)v27, v26, v28, v49, a7);
        result = (__int64)v47;
        v11 -= v45;
        v29 = v11;
        if ( a7 <= v11 )
          v29 = a7;
        if ( v29 >= v19 )
        {
          v10 = a3;
          a6 = v49;
          v9 = (__int64)v47;
          a2 = v48;
          goto LABEL_25;
        }
        if ( a7 >= v11 )
          break;
        a2 = v48;
        v20 = (__int64)v47;
      }
      v10 = a3;
      a6 = v49;
      v9 = (__int64)v47;
      a2 = v48;
    }
    v12 = v10 - a2;
    v13 = (v10 - a2) >> 4;
    if ( v10 - a2 > 0 )
    {
      v14 = a6;
      v15 = a2;
      do
      {
        v16 = *(_DWORD *)(v15 + 8);
        v14 += 16;
        v15 += 16;
        *(_DWORD *)(v14 - 8) = v16;
        *(_QWORD *)(v14 - 16) = *(_QWORD *)(v15 - 16);
        --v13;
      }
      while ( v13 );
      if ( v12 <= 0 )
        v12 = 16;
      result = a6 + v12;
      if ( v9 == a2 )
      {
        v41 = v12 >> 4;
        do
        {
          v42 = *(_DWORD *)(result - 8);
          result -= 16;
          v10 -= 16;
          *(_DWORD *)(v10 + 8) = v42;
          *(_QWORD *)v10 = *(_QWORD *)result;
          --v41;
        }
        while ( v41 );
      }
      else if ( a6 != result )
      {
        v17 = a2 - 16;
        while ( 1 )
        {
          result -= 16;
          v18 = *(_DWORD *)(v17 + 8);
          v10 -= 16;
          if ( v18 < *(_DWORD *)(result + 8) )
            break;
LABEL_16:
          *(_DWORD *)(v10 + 8) = *(_DWORD *)(result + 8);
          *(_QWORD *)v10 = *(_QWORD *)result;
          if ( a6 == result )
            return result;
        }
        while ( 1 )
        {
          *(_DWORD *)(v10 + 8) = v18;
          *(_QWORD *)v10 = *(_QWORD *)v17;
          if ( v9 == v17 )
            break;
          v17 -= 16;
          v10 -= 16;
          v18 = *(_DWORD *)(v17 + 8);
          if ( v18 >= *(_DWORD *)(result + 8) )
            goto LABEL_16;
        }
        result += 16;
        v39 = (result - a6) >> 4;
        if ( result - a6 > 0 )
        {
          do
          {
            v40 = *(_DWORD *)(result - 8);
            result -= 16;
            v10 -= 16;
            *(_DWORD *)(v10 + 8) = v40;
            *(_QWORD *)v10 = *(_QWORD *)result;
            --v39;
          }
          while ( v39 );
        }
      }
    }
  }
  return result;
}
