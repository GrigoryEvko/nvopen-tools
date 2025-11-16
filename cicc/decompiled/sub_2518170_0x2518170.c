// Function: sub_2518170
// Address: 0x2518170
//
unsigned __int64 __fastcall sub_2518170(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  unsigned __int64 *v7; // rbx
  unsigned __int64 v8; // rcx
  unsigned __int64 *v9; // r8
  __int64 v10; // rdx
  __int64 v11; // r9
  unsigned __int64 result; // rax
  unsigned __int64 v13; // rsi
  unsigned int v14; // edi
  unsigned int v15; // esi
  int v16; // eax
  unsigned __int64 *v17; // r13
  int v18; // eax
  bool v19; // zf
  __int64 v20; // r8
  __int64 v21; // r9
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rcx
  unsigned __int64 v24; // rsi
  int v25; // eax
  __int64 v26; // rcx
  unsigned __int64 *v27; // rdi
  unsigned __int64 v28; // rax
  char *v29; // rbx
  unsigned __int64 v30; // rax
  __int64 v31; // rdi
  char *v32; // rbx
  unsigned __int64 *v33; // [rsp+0h] [rbp-50h] BYREF
  unsigned __int64 *v34; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v35[8]; // [rsp+10h] [rbp-40h] BYREF

  v6 = a1;
  v7 = (unsigned __int64 *)a2;
  if ( !*(_DWORD *)(a1 + 16) )
  {
    v8 = *(_QWORD *)(a1 + 32);
    LODWORD(a1) = *(_DWORD *)(a1 + 40);
    v9 = (unsigned __int64 *)(v8 + 24LL * (unsigned int)a1);
    v10 = 0xAAAAAAAAAAAAAAABLL * ((24LL * (unsigned int)a1) >> 3);
    if ( v10 >> 2 )
    {
      v11 = 3 * (v10 >> 2);
      v10 = *(_QWORD *)(a2 + 16);
      result = v8;
      a6 = v8 + 32 * v11;
      while ( *(_QWORD *)(result + 16) != v10 )
      {
        if ( v10 == *(_QWORD *)(result + 40) )
        {
          result += 24LL;
          if ( v9 != (unsigned __int64 *)result )
            return result;
          goto LABEL_15;
        }
        if ( v10 == *(_QWORD *)(result + 64) )
        {
          result += 48LL;
          if ( v9 != (unsigned __int64 *)result )
            return result;
          goto LABEL_15;
        }
        if ( v10 == *(_QWORD *)(result + 88) )
        {
          result += 72LL;
          if ( v9 != (unsigned __int64 *)result )
            return result;
          goto LABEL_15;
        }
        result += 96LL;
        if ( a6 == result )
        {
          a6 = 0xAAAAAAAAAAAAAAABLL;
          v10 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v9 - result) >> 3);
          goto LABEL_12;
        }
      }
LABEL_9:
      if ( v9 != (unsigned __int64 *)result )
        return result;
      goto LABEL_15;
    }
    result = v8;
LABEL_12:
    switch ( v10 )
    {
      case 2LL:
        v10 = *(_QWORD *)(a2 + 16);
        break;
      case 3LL:
        v10 = *(_QWORD *)(a2 + 16);
        if ( *(_QWORD *)(result + 16) == v10 )
          goto LABEL_9;
        result += 24LL;
        break;
      case 1LL:
        v10 = *(_QWORD *)(a2 + 16);
LABEL_43:
        if ( *(_QWORD *)(result + 16) == v10 )
          goto LABEL_9;
LABEL_15:
        result = *(unsigned int *)(v6 + 44);
        v13 = (unsigned int)a1 + 1LL;
        if ( v13 > result )
        {
          a1 = v6 + 32;
          if ( v8 > (unsigned __int64)v7 || v9 <= v7 )
          {
            sub_D6B130(a1, v13, v10, v8, (__int64)v9, a6);
            LODWORD(a1) = *(_DWORD *)(v6 + 40);
            result = *(_QWORD *)(v6 + 32);
            v9 = (unsigned __int64 *)(result + 24LL * (unsigned int)a1);
          }
          else
          {
            v29 = (char *)v7 - v8;
            sub_D6B130(a1, v13, v10, v8, (__int64)v9, a6);
            result = *(_QWORD *)(v6 + 32);
            a1 = *(unsigned int *)(v6 + 40);
            v7 = (unsigned __int64 *)&v29[result];
            v9 = (unsigned __int64 *)(result + 24 * a1);
          }
        }
        if ( v9 )
        {
          *v9 = 4;
          result = v7[2];
          v9[1] = 0;
          v9[2] = result;
          if ( result != 0 && result != -4096 && result != -8192 )
            result = sub_BD6050(v9, *v7 & 0xFFFFFFFFFFFFFFF8LL);
          LODWORD(a1) = *(_DWORD *)(v6 + 40);
        }
        v14 = a1 + 1;
        *(_DWORD *)(v6 + 40) = v14;
        if ( v14 > 0x10 )
          return sub_2517F50(v6);
        return result;
      default:
        goto LABEL_15;
    }
    if ( *(_QWORD *)(result + 16) == v10 )
      goto LABEL_9;
    result += 24LL;
    goto LABEL_43;
  }
  result = sub_25116B0(a1, a2, &v33);
  if ( (_BYTE)result )
    return result;
  v15 = *(_DWORD *)(a1 + 24);
  v16 = *(_DWORD *)(a1 + 16);
  v17 = v33;
  ++*(_QWORD *)a1;
  v18 = v16 + 1;
  v34 = v17;
  if ( 4 * v18 >= 3 * v15 )
  {
    v15 *= 2;
    goto LABEL_54;
  }
  if ( v15 - *(_DWORD *)(a1 + 20) - v18 <= v15 >> 3 )
  {
LABEL_54:
    sub_2517BE0(a1, v15);
    sub_25116B0(a1, (__int64)v7, &v34);
    v17 = v34;
    v18 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v18;
  v35[2] = -4096;
  v19 = v17[2] == -4096;
  v35[0] = 4;
  v35[1] = 0;
  if ( !v19 )
    --*(_DWORD *)(a1 + 20);
  sub_D68D70(v35);
  sub_2506B90(v17, v7);
  v22 = *(unsigned int *)(a1 + 40);
  v23 = *(unsigned int *)(a1 + 44);
  v24 = v22 + 1;
  v25 = *(_DWORD *)(a1 + 40);
  if ( v22 + 1 > v23 )
  {
    v30 = *(_QWORD *)(a1 + 32);
    v31 = a1 + 32;
    if ( v30 > (unsigned __int64)v7 || (v22 = v30 + 24 * v22, (unsigned __int64)v7 >= v22) )
    {
      sub_D6B130(v31, v24, v22, v23, v20, v21);
      v22 = *(unsigned int *)(v6 + 40);
      v26 = *(_QWORD *)(v6 + 32);
      v25 = *(_DWORD *)(v6 + 40);
    }
    else
    {
      v32 = (char *)v7 - v30;
      sub_D6B130(v31, v24, v22, v23, v20, v21);
      v26 = *(_QWORD *)(v6 + 32);
      v22 = *(unsigned int *)(v6 + 40);
      v7 = (unsigned __int64 *)&v32[v26];
      v25 = *(_DWORD *)(v6 + 40);
    }
  }
  else
  {
    v26 = *(_QWORD *)(a1 + 32);
  }
  v27 = (unsigned __int64 *)(v26 + 24 * v22);
  if ( v27 )
  {
    *v27 = 4;
    v28 = v7[2];
    v27[1] = 0;
    v27[2] = v28;
    if ( v28 != -4096 && v28 != 0 && v28 != -8192 )
      sub_BD6050(v27, *v7 & 0xFFFFFFFFFFFFFFF8LL);
    v25 = *(_DWORD *)(v6 + 40);
  }
  result = (unsigned int)(v25 + 1);
  *(_DWORD *)(v6 + 40) = result;
  return result;
}
