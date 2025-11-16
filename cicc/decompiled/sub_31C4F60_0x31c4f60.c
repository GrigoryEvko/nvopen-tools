// Function: sub_31C4F60
// Address: 0x31c4f60
//
__int64 __fastcall sub_31C4F60(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // r12
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // r9
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 v11; // r8
  char *v12; // r13
  __int64 v13; // rdi
  __int64 *v14; // rcx
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  char *v17; // rdx
  __int64 v18; // r13
  __int64 *v19; // rax
  __int64 v20; // rdx
  __int64 result; // rax
  unsigned __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v25[6]; // [rsp+10h] [rbp-30h] BYREF

  v3 = a2;
  v5 = *(_QWORD *)(a1 + 8);
  v6 = *(unsigned int *)(a1 + 16);
  v24 = a2;
  v7 = sub_31C4E20(v5, v5 + 8 * v6, &v24, a3);
  v9 = *(unsigned int *)(a1 + 16);
  v10 = *(_QWORD *)(a1 + 8);
  v11 = a1 + 8;
  v12 = (char *)v7;
  v13 = 8 * v9;
  v14 = (__int64 *)(v10 + 8 * v9);
  if ( (__int64 *)v7 == v14 )
  {
    v22 = v9 + 1;
    if ( v22 > *(unsigned int *)(a1 + 20) )
    {
      sub_C8D5F0(a1 + 8, (const void *)(a1 + 24), v22, 8u, v11, v8);
      v14 = (__int64 *)(*(_QWORD *)(a1 + 8) + 8LL * *(unsigned int *)(a1 + 16));
    }
    *v14 = a2;
    ++*(_DWORD *)(a1 + 16);
  }
  else
  {
    LODWORD(v15) = *(_DWORD *)(a1 + 16);
    v16 = v9 + 1;
    if ( v16 > *(unsigned int *)(a1 + 20) )
    {
      sub_C8D5F0(a1 + 8, (const void *)(a1 + 24), v16, 8u, v11, v8);
      v23 = *(_QWORD *)(a1 + 8);
      v15 = *(unsigned int *)(a1 + 16);
      v13 = 8 * v15;
      v12 = &v12[v23 - v10];
      v10 = v23;
      v14 = (__int64 *)(v23 + 8 * v15);
    }
    v17 = (char *)(v10 + v13 - 8);
    if ( v14 )
    {
      *v14 = *(_QWORD *)v17;
      v10 = *(_QWORD *)(a1 + 8);
      v15 = *(unsigned int *)(a1 + 16);
      v13 = 8 * v15;
      v17 = (char *)(v10 + 8 * v15 - 8);
    }
    if ( v12 != v17 )
    {
      memmove((void *)(v10 + v13 - (v17 - v12)), v12, v17 - v12);
      LODWORD(v15) = *(_DWORD *)(a1 + 16);
    }
    *(_DWORD *)(a1 + 16) = v15 + 1;
    *(_QWORD *)v12 = a2;
  }
  v18 = sub_B43CA0(*(_QWORD *)(a2 + 16)) + 312;
  if ( sub_318B630(a2) && (*(_DWORD *)(a2 + 8) != 37 || sub_318B6C0(a2)) )
  {
    if ( sub_318B670(a2) )
    {
      v3 = sub_318B680(a2);
    }
    else if ( *(_DWORD *)(a2 + 8) == 37 )
    {
      v3 = sub_318B6C0(a2);
    }
  }
  v19 = sub_318EB80(v3);
  v25[0] = sub_9208B0(v18, *v19);
  v25[1] = v20;
  result = sub_CA1930(v25);
  *(_DWORD *)(a1 + 148) += result;
  return result;
}
