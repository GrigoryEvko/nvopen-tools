// Function: sub_B33310
// Address: 0xb33310
//
__int64 __fastcall sub_B33310(
        unsigned int **a1,
        unsigned __int64 a2,
        int a3,
        int a4,
        int a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10)
{
  int v10; // ebx
  __int64 v14; // r9
  __int64 v15; // rsi
  int v16; // edi
  __int64 v17; // rax
  unsigned __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // r12
  unsigned int *v21; // rbx
  __int64 v22; // r13
  __int64 v23; // rdx
  __int64 v24; // rsi
  __int64 *v26; // rax
  unsigned int v27; // [rsp+24h] [rbp-8Ch]
  char v31; // [rsp+50h] [rbp-60h] BYREF
  __int16 v32; // [rsp+70h] [rbp-40h]

  v32 = 257;
  v14 = a9 + 56 * a10;
  if ( v14 == a9 )
  {
    v16 = 0;
  }
  else
  {
    v15 = a9;
    v16 = 0;
    do
    {
      v17 = *(_QWORD *)(v15 + 40) - *(_QWORD *)(v15 + 32);
      v15 += 56;
      v16 += v17 >> 3;
    }
    while ( v14 != v15 );
  }
  v27 = v16 + a8 + 3;
  v18 = ((unsigned __int64)(unsigned int)(16 * a10) << 32) | v27;
  v20 = sub_BD2CC0(88, v18);
  if ( v20 )
  {
    LOBYTE(v10) = 16 * (_DWORD)a10 != 0;
    sub_B44260(v20, **(_QWORD **)(a2 + 16), 5, (v10 << 28) | v27 & 0x7FFFFFF, 0, 0);
    v18 = a2;
    *(_QWORD *)(v20 + 72) = 0;
    sub_B4A9C0(v20, a2, a3, a4, a5, (unsigned int)&v31, a7, a8, a9, a10);
  }
  if ( *((_BYTE *)a1 + 108) )
  {
    v26 = (__int64 *)sub_BD5C60(v20, v18, v19);
    *(_QWORD *)(v20 + 72) = sub_A7A090((__int64 *)(v20 + 72), v26, -1, 72);
  }
  (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v20,
    a6,
    a1[7],
    a1[8]);
  v21 = *a1;
  v22 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  if ( *a1 != (unsigned int *)v22 )
  {
    do
    {
      v23 = *((_QWORD *)v21 + 1);
      v24 = *v21;
      v21 += 4;
      sub_B99FD0(v20, v24, v23);
    }
    while ( (unsigned int *)v22 != v21 );
  }
  return v20;
}
