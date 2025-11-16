// Function: sub_2D598D0
// Address: 0x2d598d0
//
__int64 __fastcall sub_2D598D0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r12
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned __int64 v15; // rdi
  char *v16; // rbx
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // rsi
  int v19; // eax
  _QWORD *v20; // rdx
  __int64 result; // rax
  char *v22; // rbx
  _QWORD v23[7]; // [rsp+8h] [rbp-38h] BYREF

  v7 = sub_22077B0(0x20u);
  v10 = v7;
  if ( v7 )
  {
    *(_QWORD *)(v7 + 8) = a2;
    *(_QWORD *)v7 = off_49D4030;
    *(_DWORD *)(v7 + 24) = a3;
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v11 = *(_QWORD *)(a2 - 8) + 32LL * a3;
    else
      v11 = a2 + 32 * (a3 - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    *(_QWORD *)(v7 + 16) = *(_QWORD *)v11;
    if ( *(_QWORD *)v11 )
    {
      v12 = *(_QWORD *)(v11 + 8);
      **(_QWORD **)(v11 + 16) = v12;
      if ( v12 )
        *(_QWORD *)(v12 + 16) = *(_QWORD *)(v11 + 16);
    }
    *(_QWORD *)v11 = a4;
    if ( a4 )
    {
      v13 = *(_QWORD *)(a4 + 16);
      *(_QWORD *)(v11 + 8) = v13;
      if ( v13 )
        *(_QWORD *)(v13 + 16) = v11 + 8;
      *(_QWORD *)(v11 + 16) = a4 + 16;
      *(_QWORD *)(a4 + 16) = v11;
    }
  }
  v14 = *(unsigned int *)(a1 + 8);
  v15 = *(unsigned int *)(a1 + 12);
  v23[0] = v10;
  v16 = (char *)v23;
  v17 = *(_QWORD *)a1;
  v18 = v14 + 1;
  v19 = v14;
  if ( v14 + 1 > v15 )
  {
    if ( v17 > (unsigned __int64)v23 || (unsigned __int64)v23 >= v17 + 8 * v14 )
    {
      sub_2D57B00(a1, v18, v14, v17, v8, v9);
      v14 = *(unsigned int *)(a1 + 8);
      v17 = *(_QWORD *)a1;
      v19 = *(_DWORD *)(a1 + 8);
    }
    else
    {
      v22 = (char *)v23 - v17;
      sub_2D57B00(a1, v18, v14, v17, v8, v9);
      v17 = *(_QWORD *)a1;
      v14 = *(unsigned int *)(a1 + 8);
      v16 = &v22[*(_QWORD *)a1];
      v19 = *(_DWORD *)(a1 + 8);
    }
  }
  v20 = (_QWORD *)(v17 + 8 * v14);
  if ( v20 )
  {
    *v20 = *(_QWORD *)v16;
    *(_QWORD *)v16 = 0;
    v10 = v23[0];
    v19 = *(_DWORD *)(a1 + 8);
  }
  result = (unsigned int)(v19 + 1);
  *(_DWORD *)(a1 + 8) = result;
  if ( v10 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 8LL))(v10);
  return result;
}
