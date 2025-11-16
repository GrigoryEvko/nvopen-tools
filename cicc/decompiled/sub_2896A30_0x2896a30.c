// Function: sub_2896A30
// Address: 0x2896a30
//
__int64 __fastcall sub_2896A30(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rsi
  __int64 v6; // rdx
  __int64 v7; // rdi
  unsigned int v8; // ecx
  __int64 *v9; // rax
  __int64 v10; // r10
  __int64 v11; // rdx
  __int64 v12; // rbx
  unsigned __int64 v13; // rsi
  _BYTE *v14; // rax
  int v16; // eax
  int v17; // r11d

  v5 = *(_QWORD *)(a1 + 112);
  v6 = *(unsigned int *)(v5 + 24);
  v7 = *(_QWORD *)(v5 + 8);
  if ( !(_DWORD)v6 )
    return sub_904010(a3, "unknown");
  v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (__int64 *)(v7 + 16LL * v8);
  v10 = *v9;
  if ( *v9 != a2 )
  {
    v16 = 1;
    while ( v10 != -4096 )
    {
      v17 = v16 + 1;
      v8 = (v6 - 1) & (v16 + v8);
      v9 = (__int64 *)(v7 + 16LL * v8);
      v10 = *v9;
      if ( *v9 == a2 )
        goto LABEL_3;
      v16 = v17;
    }
    return sub_904010(a3, "unknown");
  }
LABEL_3:
  if ( v9 == (__int64 *)(v7 + 16 * v6) )
    return sub_904010(a3, "unknown");
  v11 = *(_QWORD *)(v5 + 32);
  v12 = v11 + 176LL * *((unsigned int *)v9 + 2);
  if ( v12 == 176LL * *(unsigned int *)(v5 + 40) + v11 )
    return sub_904010(a3, "unknown");
  if ( *(_BYTE *)(v12 + 168) )
    v13 = *(unsigned int *)(*(_QWORD *)(**(_QWORD **)(v12 + 8) + 8LL) + 32LL);
  else
    v13 = *(unsigned int *)(v12 + 16);
  sub_CB59D0(a3, v13);
  v14 = *(_BYTE **)(a3 + 32);
  if ( *(_BYTE **)(a3 + 24) == v14 )
  {
    sub_CB6200(a3, (unsigned __int8 *)"x", 1u);
  }
  else
  {
    *v14 = 120;
    ++*(_QWORD *)(a3 + 32);
  }
  if ( *(_BYTE *)(v12 + 168) )
    return sub_CB59D0(a3, *(unsigned int *)(v12 + 16));
  else
    return sub_CB59D0(a3, *(unsigned int *)(*(_QWORD *)(**(_QWORD **)(v12 + 8) + 8LL) + 32LL));
}
