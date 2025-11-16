// Function: sub_2A481C0
// Address: 0x2a481c0
//
__int64 __fastcall sub_2A481C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rbx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 result; // rax
  __int64 v14; // rax

  v8 = sub_2A47E20(a1, a3, a3, a4, a5, a6);
  if ( !*(_DWORD *)(v8 + 8) )
  {
    v14 = *(unsigned int *)(a2 + 8);
    if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
      sub_C8D5F0(a2, (const void *)(a2 + 16), v14 + 1, 8u, v9, v10);
      v14 = *(unsigned int *)(a2 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a2 + 8 * v14) = a3;
    ++*(_DWORD *)(a2 + 8);
  }
  v11 = *(_QWORD *)a1;
  v12 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
  *(_QWORD *)(a4 + 16) = *(_QWORD *)a1 + 8LL;
  v12 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(a4 + 8) = v12 | *(_QWORD *)(a4 + 8) & 7LL;
  *(_QWORD *)(v12 + 8) = a4 + 8;
  *(_QWORD *)(v11 + 8) = *(_QWORD *)(v11 + 8) & 7LL | (a4 + 8);
  result = *(unsigned int *)(v8 + 8);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(v8 + 12) )
  {
    sub_C8D5F0(v8, (const void *)(v8 + 16), result + 1, 8u, v9, v10);
    result = *(unsigned int *)(v8 + 8);
  }
  *(_QWORD *)(*(_QWORD *)v8 + 8 * result) = a4;
  ++*(_DWORD *)(v8 + 8);
  return result;
}
