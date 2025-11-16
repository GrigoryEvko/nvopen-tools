// Function: sub_32B0350
// Address: 0x32b0350
//
unsigned __int64 __fastcall sub_32B0350(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rbx
  __int64 v9; // rcx
  __int64 *v10; // rdx
  __int64 v11; // rax
  int v12; // esi
  int v13; // edx
  __int64 v14; // rdi
  unsigned __int64 result; // rax
  __int64 v16; // rcx
  __int64 v17; // rdx
  unsigned __int64 v19; // [rsp+8h] [rbp-28h]

  v8 = *(_QWORD *)a1;
  v9 = *(unsigned int *)(*(_QWORD *)a1 + 140LL);
  if ( (_DWORD)v9 )
  {
    v10 = (__int64 *)(v8 + 72);
    v11 = 0;
    do
    {
      if ( a2 < *v10 )
        break;
      v11 = (unsigned int)(v11 + 1);
      ++v10;
    }
    while ( (_DWORD)v9 != (_DWORD)v11 );
  }
  else
  {
    v11 = 0;
  }
  v12 = *(_DWORD *)(v8 + 136);
  v13 = *(_DWORD *)(a1 + 20);
  *(_DWORD *)(a1 + 16) = 0;
  v14 = a1 + 8;
  if ( v12 )
    v8 += 8;
  result = v9 | (v11 << 32);
  v16 = 0;
  if ( !v13 )
  {
    v19 = result;
    sub_C8D5F0(v14, (const void *)(a1 + 24), 1u, 0x10u, a5, a6);
    result = v19;
    v16 = 16LL * *(unsigned int *)(a1 + 16);
  }
  v17 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(v17 + v16) = v8;
  *(_QWORD *)(v17 + v16 + 8) = result;
  if ( (*(_DWORD *)(a1 + 16))++ != -1 )
  {
    result = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(result + 12) < *(_DWORD *)(result + 8) )
      return sub_32B01E0(a1, a2, v17, v16, a5, a6);
  }
  return result;
}
