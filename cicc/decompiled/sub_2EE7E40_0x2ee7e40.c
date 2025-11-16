// Function: sub_2EE7E40
// Address: 0x2ee7e40
//
unsigned __int64 __fastcall sub_2EE7E40(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 (*v5)(); // rdx
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // r12
  int v14; // r13d
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 i; // rdx
  __int64 v18; // rdx
  unsigned int v19; // r12d
  unsigned __int64 result; // rax
  __int64 v21; // rdx
  __int64 j; // rdx

  *a1 = a2;
  v4 = *(_QWORD *)(a2 + 16);
  v5 = *(__int64 (**)())(*(_QWORD *)v4 + 128LL);
  v6 = 0;
  if ( v5 != sub_2DAC790 )
    v6 = ((__int64 (__fastcall *)(_QWORD))v5)(*(_QWORD *)(a2 + 16));
  a1[1] = v6;
  a1[2] = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 200LL))(v4);
  v7 = *(_QWORD *)(*a1 + 32);
  a1[4] = a3;
  a1[3] = v7;
  sub_2FF7BB0(a1 + 5, v4);
  v10 = *a1;
  v11 = *((unsigned int *)a1 + 86);
  v12 = (__int64)(*(_QWORD *)(*a1 + 104) - *(_QWORD *)(*a1 + 96)) >> 3;
  v13 = (unsigned int)v12;
  if ( (unsigned int)v12 != v11 )
  {
    v14 = v12;
    if ( (unsigned int)v12 < v11 )
    {
      *((_DWORD *)a1 + 86) = v12;
      v12 = (__int64)(*(_QWORD *)(v10 + 104) - *(_QWORD *)(v10 + 96)) >> 3;
    }
    else
    {
      if ( (unsigned int)v12 > (unsigned __int64)*((unsigned int *)a1 + 87) )
      {
        sub_C8D5F0(
          (__int64)(a1 + 42),
          a1 + 44,
          (unsigned int)((__int64)(*(_QWORD *)(*a1 + 104) - *(_QWORD *)(*a1 + 96)) >> 3),
          8u,
          v8,
          v9);
        v11 = *((unsigned int *)a1 + 86);
      }
      v15 = a1[42];
      v16 = v15 + 8 * v11;
      for ( i = v15 + 8 * v13; i != v16; v16 += 8 )
      {
        if ( v16 )
        {
          *(_BYTE *)(v16 + 4) = 0;
          *(_DWORD *)v16 = -1;
        }
      }
      v18 = *a1;
      *((_DWORD *)a1 + 86) = v14;
      v12 = (__int64)(*(_QWORD *)(v18 + 104) - *(_QWORD *)(v18 + 96)) >> 3;
    }
  }
  v19 = v12 * *((_DWORD *)a1 + 22);
  result = *((unsigned int *)a1 + 98);
  if ( v19 != result )
  {
    if ( v19 >= result )
    {
      if ( v19 > (unsigned __int64)*((unsigned int *)a1 + 99) )
      {
        sub_C8D5F0((__int64)(a1 + 48), a1 + 50, v19, 4u, v8, v9);
        result = *((unsigned int *)a1 + 98);
      }
      v21 = a1[48];
      result = v21 + 4 * result;
      for ( j = v21 + 4LL * v19; j != result; result += 4LL )
      {
        if ( result )
          *(_DWORD *)result = 0;
      }
    }
    *((_DWORD *)a1 + 98) = v19;
  }
  return result;
}
