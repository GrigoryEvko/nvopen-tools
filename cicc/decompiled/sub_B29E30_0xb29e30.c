// Function: sub_B29E30
// Address: 0xb29e30
//
__int64 __fastcall sub_B29E30(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  __int64 v7; // rcx
  unsigned int v8; // eax
  unsigned int v9; // edx
  __int64 *v10; // r9
  __int64 v11; // rcx
  unsigned int v12; // eax
  __int64 v13; // rcx
  __int64 result; // rax
  __int64 *v15; // rbx
  __int64 *v16; // r13
  __int64 v17; // rsi
  unsigned int v18; // eax
  __int64 *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // [rsp+8h] [rbp-C8h]
  __int64 *v24; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v25; // [rsp+18h] [rbp-B8h]
  _BYTE v26[176]; // [rsp+20h] [rbp-B0h] BYREF

  v4 = a2;
  if ( a3 )
  {
    v7 = (unsigned int)(*(_DWORD *)(a3 + 44) + 1);
    v8 = *(_DWORD *)(a3 + 44) + 1;
  }
  else
  {
    v7 = 0;
    v8 = 0;
  }
  v9 = *(_DWORD *)(a1 + 56);
  if ( v8 >= v9 || (v10 = *(__int64 **)(*(_QWORD *)(a1 + 48) + 8 * v7)) == 0 )
  {
    v22 = 0;
    if ( v9 )
      v22 = **(_QWORD **)(a1 + 48);
    v23 = sub_B1BBB0(a1, a3, v22);
    sub_B1A4E0(a1, a3);
    v9 = *(_DWORD *)(a1 + 56);
    *(_BYTE *)(a1 + 136) = 0;
    v10 = (__int64 *)v23;
    if ( a4 )
      goto LABEL_6;
LABEL_21:
    v11 = 0;
    v12 = 0;
    goto LABEL_7;
  }
  *(_BYTE *)(a1 + 136) = 0;
  if ( !a4 )
    goto LABEL_21;
LABEL_6:
  v11 = (unsigned int)(*(_DWORD *)(a4 + 44) + 1);
  v12 = *(_DWORD *)(a4 + 44) + 1;
LABEL_7:
  if ( v12 < v9 )
  {
    v13 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v11);
    if ( v13 )
      return sub_B29620((_QWORD **)a1, a2, v10, v13);
  }
  v24 = (__int64 *)v26;
  v25 = 0x800000000LL;
  result = sub_B21850(a1, a2, a4, v10, (__int64)&v24);
  v15 = v24;
  v16 = &v24[2 * (unsigned int)v25];
  if ( v24 != v16 )
  {
    do
    {
      v20 = *v15;
      v21 = v15[1];
      if ( *v15 )
      {
        v17 = (unsigned int)(*(_DWORD *)(v20 + 44) + 1);
        v18 = *(_DWORD *)(v20 + 44) + 1;
      }
      else
      {
        v17 = 0;
        v18 = 0;
      }
      v19 = 0;
      if ( v18 < *(_DWORD *)(a1 + 56) )
        v19 = *(__int64 **)(*(_QWORD *)(a1 + 48) + 8 * v17);
      a2 = v4;
      v15 += 2;
      result = sub_B29620((_QWORD **)a1, v4, v19, v21);
    }
    while ( v16 != v15 );
    v16 = v24;
  }
  if ( v16 != (__int64 *)v26 )
    return _libc_free(v16, a2);
  return result;
}
