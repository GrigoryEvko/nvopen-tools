// Function: sub_C6B410
// Address: 0xc6b410
//
__int64 __fastcall sub_C6B410(__int64 a1, unsigned __int8 *a2, unsigned __int64 a3)
{
  unsigned __int64 v5; // rcx
  unsigned __int64 v6; // rdx
  int v7; // eax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 result; // rax
  __int64 v12; // rdi
  _BYTE *v13; // rax
  unsigned __int8 *v14[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v15; // [rsp+10h] [rbp-30h] BYREF

  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - 4) )
  {
    v12 = *(_QWORD *)(a1 + 160);
    v13 = *(_BYTE **)(v12 + 32);
    if ( (unsigned __int64)v13 >= *(_QWORD *)(v12 + 24) )
    {
      sub_CB5D20(v12, 44);
    }
    else
    {
      *(_QWORD *)(v12 + 32) = v13 + 1;
      *v13 = 44;
    }
  }
  sub_C6A6A0(a1);
  sub_C6A6F0(a1);
  *(_BYTE *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - 4) = 1;
  v5 = *(unsigned int *)(a1 + 8);
  v6 = *(unsigned int *)(a1 + 12);
  v7 = *(_DWORD *)(a1 + 8);
  if ( v5 >= v6 )
  {
    if ( v6 < v5 + 1 )
    {
      sub_C8D5F0(a1, a1 + 16, v5 + 1, 8);
      v5 = *(unsigned int *)(a1 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a1 + 8 * v5) = 0;
    v8 = *(_QWORD *)a1;
    v10 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
    *(_DWORD *)(a1 + 8) = v10;
  }
  else
  {
    v8 = *(_QWORD *)a1;
    v9 = *(_QWORD *)a1 + 8 * v5;
    if ( v9 )
    {
      *(_DWORD *)v9 = 0;
      *(_BYTE *)(v9 + 4) = 0;
      v7 = *(_DWORD *)(a1 + 8);
      v8 = *(_QWORD *)a1;
    }
    v10 = (unsigned int)(v7 + 1);
    *(_DWORD *)(a1 + 8) = v10;
  }
  *(_DWORD *)(v8 + 8 * v10 - 8) = 0;
  if ( (unsigned __int8)sub_C6A630((char *)a2, a3, 0) )
  {
    sub_C69320(*(_QWORD *)(a1 + 160), a2, a3);
  }
  else
  {
    sub_C6B0E0((__int64 *)v14, (__int64)a2, a3);
    sub_C69320(*(_QWORD *)(a1 + 160), v14[0], (__int64)v14[1]);
    if ( (__int64 *)v14[0] != &v15 )
      j_j___libc_free_0(v14[0], v15 + 1);
  }
  sub_CB5D20(*(_QWORD *)(a1 + 160), 58);
  result = *(unsigned int *)(a1 + 168);
  if ( (_DWORD)result )
    return sub_CB5D20(*(_QWORD *)(a1 + 160), 32);
  return result;
}
