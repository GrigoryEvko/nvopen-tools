// Function: sub_2FF67E0
// Address: 0x2ff67e0
//
__int64 __fastcall sub_2FF67E0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, __int64 a5, __int64 a6)
{
  void *v9; // rdi
  __int64 v10; // r8
  unsigned int v11; // ebx
  unsigned __int16 ***v12; // rsi
  __int64 v13; // rdi
  unsigned int v14; // esi
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rdx
  _QWORD *v18; // rcx
  unsigned __int16 ****v20; // r15
  unsigned __int16 ****i; // rbx
  int v22; // [rsp+4h] [rbp-3Ch]
  int v23; // [rsp+8h] [rbp-38h]

  v9 = (void *)(a1 + 16);
  v10 = *(unsigned int *)(a2 + 16);
  *(_QWORD *)a1 = v9;
  *(_QWORD *)(a1 + 8) = 0x600000000LL;
  v11 = (unsigned int)(v10 + 63) >> 6;
  if ( v11 > 6 )
  {
    v22 = v10;
    sub_C8D5F0(a1, v9, v11, 8u, v10, a6);
    memset(*(void **)a1, 0, 8LL * v11);
    *(_DWORD *)(a1 + 8) = v11;
    LODWORD(v10) = v22;
  }
  else
  {
    if ( v11 )
    {
      v23 = v10;
      memset(v9, 0, (size_t)v9 + 8 * v11 - a1 - 16);
      LODWORD(v10) = v23;
    }
    *(_DWORD *)(a1 + 8) = v11;
  }
  *(_DWORD *)(a1 + 64) = v10;
  if ( a4 )
  {
    v12 = (unsigned __int16 ***)sub_2FF6410(a2, a4);
    if ( v12 )
      sub_2FF5520(a3, v12, (_QWORD *)a1);
  }
  else
  {
    v20 = *(unsigned __int16 *****)(a2 + 288);
    for ( i = *(unsigned __int16 *****)(a2 + 280); v20 != i; ++i )
    {
      if ( *((_BYTE *)**i + 29) )
        sub_2FF5520(a3, *i, (_QWORD *)a1);
    }
  }
  v13 = *(_QWORD *)(a3 + 32);
  v14 = *(_DWORD *)(v13 + 392);
  if ( *(_DWORD *)(a1 + 8) <= v14 )
    v14 = *(_DWORD *)(a1 + 8);
  if ( v14 )
  {
    v15 = 0;
    v16 = 8LL * v14;
    do
    {
      v17 = *(_QWORD *)(*(_QWORD *)(v13 + 384) + v15);
      v18 = (_QWORD *)(v15 + *(_QWORD *)a1);
      v15 += 8;
      *v18 &= ~v17;
    }
    while ( v15 != v16 );
  }
  return a1;
}
