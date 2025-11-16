// Function: sub_31C2860
// Address: 0x31c2860
//
__int64 __fastcall sub_31C2860(__int64 a1, char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r8
  __int64 v10; // rdx
  __int64 v11; // r14
  unsigned __int64 v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 *v16; // rcx
  char *v17; // rdx
  __int64 v18; // r13
  __int64 *v19; // rax
  __int64 v20; // rdx
  __int64 result; // rax
  __int64 v22; // rdx
  _QWORD v23[6]; // [rsp+0h] [rbp-30h] BYREF

  v6 = a1 + 8;
  v10 = *(unsigned int *)(a1 + 16);
  v11 = *(_QWORD *)(a1 + 8);
  v12 = *(unsigned int *)(a1 + 20);
  v13 = 8 * v10;
  LODWORD(v14) = v10;
  v15 = v10 + 1;
  v16 = (__int64 *)(v11 + v13);
  if ( a2 == (char *)(v11 + v13) )
  {
    if ( v15 > v12 )
    {
      sub_C8D5F0(v6, (const void *)(a1 + 24), v15, 8u, v6, a6);
      v16 = (__int64 *)(*(_QWORD *)(a1 + 8) + 8LL * *(unsigned int *)(a1 + 16));
    }
    *v16 = a3;
    ++*(_DWORD *)(a1 + 16);
  }
  else
  {
    if ( v15 > v12 )
    {
      sub_C8D5F0(v6, (const void *)(a1 + 24), v15, 8u, v6, a6);
      v22 = *(_QWORD *)(a1 + 8);
      v14 = *(unsigned int *)(a1 + 16);
      v13 = 8 * v14;
      a2 = &a2[v22 - v11];
      v11 = v22;
      v16 = (__int64 *)(v22 + 8 * v14);
    }
    v17 = (char *)(v11 + v13 - 8);
    if ( v16 )
    {
      *v16 = *(_QWORD *)v17;
      v11 = *(_QWORD *)(a1 + 8);
      v14 = *(unsigned int *)(a1 + 16);
      v13 = 8 * v14;
      v17 = (char *)(v11 + 8 * v14 - 8);
    }
    if ( a2 != v17 )
    {
      memmove((void *)(v11 + v13 - (v17 - a2)), a2, v17 - a2);
      LODWORD(v14) = *(_DWORD *)(a1 + 16);
    }
    *(_DWORD *)(a1 + 16) = v14 + 1;
    *(_QWORD *)a2 = a3;
  }
  v18 = sub_B43CA0(*(_QWORD *)(a3 + 16)) + 312;
  if ( sub_318B630(a3) && (*(_DWORD *)(a3 + 8) != 37 || sub_318B6C0(a3)) )
  {
    if ( sub_318B670(a3) )
    {
      a3 = sub_318B680(a3);
    }
    else if ( *(_DWORD *)(a3 + 8) == 37 )
    {
      a3 = sub_318B6C0(a3);
    }
  }
  v19 = sub_318EB80(a3);
  v23[0] = sub_9208B0(v18, *v19);
  v23[1] = v20;
  result = sub_CA1930(v23);
  *(_DWORD *)(a1 + 148) += result;
  return result;
}
