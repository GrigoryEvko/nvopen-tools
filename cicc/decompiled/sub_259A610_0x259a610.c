// Function: sub_259A610
// Address: 0x259a610
//
__int64 __fastcall sub_259A610(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4)
{
  unsigned int v7; // eax
  __int64 v8; // r8
  __int64 v9; // r9
  unsigned int v10; // r12d
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v17; // rsi
  __int64 v18; // rsi
  __int64 v19; // [rsp-10h] [rbp-60h]
  __int64 v20; // [rsp-8h] [rbp-58h]
  unsigned __int8 v21; // [rsp+Fh] [rbp-41h] BYREF
  __m128i v22[4]; // [rsp+10h] [rbp-40h] BYREF

  sub_250D230((unsigned __int64 *)v22, (unsigned __int64)a2, 5, 0);
  v7 = sub_2599C30(a1, a3, v22, 1, &v21, 0, 0);
  if ( (_BYTE)v7 )
  {
    return v21 ^ 1u;
  }
  else
  {
    v10 = v7;
    if ( (unsigned int)*a2 - 30 > 0xA )
    {
      v17 = *((_QWORD *)a2 + 4);
      if ( v17 == *((_QWORD *)a2 + 5) + 48LL || !v17 )
        v18 = 0;
      else
        v18 = v17 - 24;
      sub_25594A0(a4, v18, v19, v20, v8, v9);
    }
    else
    {
      v11 = *(_QWORD *)(sub_B46EC0((__int64)a2, 0) + 56);
      v14 = v11 - 24;
      if ( !v11 )
        v14 = 0;
      v15 = *(unsigned int *)(a4 + 8);
      if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
      {
        sub_C8D5F0(a4, (const void *)(a4 + 16), v15 + 1, 8u, v12, v13);
        v15 = *(unsigned int *)(a4 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a4 + 8 * v15) = v14;
      ++*(_DWORD *)(a4 + 8);
    }
  }
  return v10;
}
