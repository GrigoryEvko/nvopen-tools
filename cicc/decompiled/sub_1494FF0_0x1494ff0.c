// Function: sub_1494FF0
// Address: 0x1494ff0
//
bool __fastcall sub_1494FF0(__int64 a1, __int64 a2, int a3, __m128i a4, __m128i a5)
{
  __int64 *v7; // rax
  int v8; // eax
  __int64 v9; // rdx
  int v10; // eax
  __int64 v12; // rsi
  unsigned int v13; // edi
  __int64 v14; // rcx
  __int64 v15; // r9
  int v16; // ecx
  int v17; // r10d

  v7 = sub_1494E70(a1, a2, a4, a5);
  v8 = sub_1479390((__int64)v7, *(_QWORD *)(a1 + 112));
  v9 = *(unsigned int *)(a1 + 56);
  v10 = a3 & ~v8;
  if ( !(_DWORD)v9 )
    return v10 == 0;
  v12 = *(_QWORD *)(a1 + 40);
  v13 = (v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v14 = v12 + 48LL * v13;
  v15 = *(_QWORD *)(v14 + 24);
  if ( v15 != a2 )
  {
    v16 = 1;
    while ( v15 != -8 )
    {
      v17 = v16 + 1;
      v13 = (v9 - 1) & (v16 + v13);
      v14 = v12 + 48LL * v13;
      v15 = *(_QWORD *)(v14 + 24);
      if ( v15 == a2 )
        goto LABEL_4;
      v16 = v17;
    }
    return v10 == 0;
  }
LABEL_4:
  if ( v14 == 48 * v9 + v12 )
    return v10 == 0;
  return (~*(_DWORD *)(v14 + 40) & v10) == 0;
}
