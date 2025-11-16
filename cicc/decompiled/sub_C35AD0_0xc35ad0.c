// Function: sub_C35AD0
// Address: 0xc35ad0
//
char *__fastcall sub_C35AD0(__int64 a1, __int64 *a2, unsigned int a3, unsigned int a4, char a5)
{
  char v6; // al
  __int64 v9; // rax
  unsigned int v10; // ecx
  char v11; // r8
  char *result; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // rcx
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rdx
  char *v18; // rax
  char *v19; // rdi
  __int64 v20; // rbx
  __int64 v21; // rbx
  char *v22; // [rsp-10h] [rbp-70h]
  int v23; // [rsp+0h] [rbp-60h]
  int v24; // [rsp+Ch] [rbp-54h]
  _QWORD *v25; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v26; // [rsp+18h] [rbp-48h]
  _QWORD *v27; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v28; // [rsp+28h] [rbp-38h]

  v6 = *(_BYTE *)(a1 + 20) & 7;
  if ( v6 == 1 )
  {
    v13 = a2[1];
    if ( v13 + 3 > (unsigned __int64)a2[2] )
    {
      sub_C8D290(a2, a2 + 3, v13 + 3, 1);
      v13 = a2[1];
    }
    result = (char *)(*a2 + v13);
    *(_WORD *)result = 24910;
    result[2] = 78;
    a2[1] += 3;
  }
  else if ( v6 == 3 )
  {
    result = (char *)a2[1];
    if ( (*(_BYTE *)(a1 + 20) & 8) != 0 )
    {
      if ( a2[2] < (unsigned __int64)(result + 1) )
      {
        sub_C8D290(a2, a2 + 3, result + 1, 1);
        result = (char *)a2[1];
      }
      result[*a2] = 45;
      result = (char *)(a2[1] + 1);
      a2[1] = (__int64)result;
    }
    v17 = a2[2];
    if ( a4 )
    {
      if ( v17 < (unsigned __int64)(result + 1) )
      {
        sub_C8D290(a2, a2 + 3, result + 1, 1);
        result = (char *)a2[1];
      }
      result[*a2] = 48;
      ++a2[1];
    }
    else if ( a5 )
    {
      if ( v17 < (unsigned __int64)(result + 6) )
      {
        sub_C8D290(a2, a2 + 3, result + 6, 1);
        result = (char *)a2[1];
      }
      result += *a2;
      *(_DWORD *)result = 1160785456;
      *((_WORD *)result + 2) = 12331;
      a2[1] += 6;
    }
    else
    {
      if ( v17 < (unsigned __int64)(result + 3) )
      {
        sub_C8D290(a2, a2 + 3, result + 3, 1);
        result = (char *)a2[1];
      }
      v18 = &result[*a2];
      *(_WORD *)v18 = 11824;
      v18[2] = 48;
      result = (char *)a2[1];
      v19 = result + 3;
      a2[1] = (__int64)(result + 3);
      if ( a3 > 1 )
      {
        v20 = a3 - 1;
        if ( (unsigned __int64)&v19[v20] > a2[2] )
        {
          sub_C8D290(a2, a2 + 3, &v19[v20], 1);
          v19 = (char *)a2[1];
        }
        result = (char *)memset(&v19[*a2], 48, a3 - 1);
        v21 = a2[1] + v20;
        a2[1] = v21;
        v19 = (char *)v21;
      }
      if ( (unsigned __int64)(v19 + 4) > a2[2] )
      {
        result = (char *)sub_C8D290(a2, a2 + 3, v19 + 4, 1);
        v19 = (char *)a2[1];
      }
      *(_DWORD *)&v19[*a2] = 808463205;
      a2[1] += 4;
    }
  }
  else if ( v6 )
  {
    v23 = *(_DWORD *)(*(_QWORD *)a1 + 8LL);
    v24 = *(_DWORD *)(a1 + 16) - (v23 - 1);
    v9 = sub_C33930(a1);
    v10 = (unsigned int)(v23 + 63) >> 6;
    if ( !v10 )
      v10 = 1;
    sub_C438C0(&v25, *(unsigned int *)(*(_QWORD *)a1 + 8LL), v9, v10);
    v11 = a5;
    v28 = v26;
    if ( v26 > 0x40 )
    {
      sub_C43780(&v27, &v25);
      v11 = a5;
    }
    else
    {
      v27 = v25;
    }
    sub_C32110(a2, (*(_BYTE *)(a1 + 20) & 8) != 0, v24, &v27, a3, a4, v11);
    result = v22;
    if ( v28 > 0x40 && v27 )
      result = (char *)j_j___libc_free_0_0(v27);
    if ( v26 > 0x40 )
    {
      if ( v25 )
        return (char *)j_j___libc_free_0_0(v25);
    }
  }
  else
  {
    v14 = a2[1];
    v15 = a2[2];
    v16 = v14 + 4;
    if ( (*(_BYTE *)(a1 + 20) & 8) != 0 )
    {
      if ( v16 > v15 )
      {
        sub_C8D290(a2, a2 + 3, v16, 1);
        v14 = a2[1];
      }
      result = (char *)(*a2 + v14);
      *(_DWORD *)result = 1718503725;
      a2[1] += 4;
    }
    else
    {
      if ( v16 > v15 )
      {
        sub_C8D290(a2, a2 + 3, v16, 1);
        v14 = a2[1];
      }
      result = (char *)(*a2 + v14);
      *(_DWORD *)result = 1718503723;
      a2[1] += 4;
    }
  }
  return result;
}
