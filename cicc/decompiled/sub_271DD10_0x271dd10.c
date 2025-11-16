// Function: sub_271DD10
// Address: 0x271dd10
//
__int64 __fastcall sub_271DD10(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4, unsigned int a5, __int64 a6)
{
  int v10; // eax
  unsigned int v11; // r12d
  unsigned __int8 v12; // al
  _QWORD *v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  _QWORD *v18; // rax
  int v19; // eax
  __int64 v20; // rsi
  int v21; // ecx
  unsigned int v22; // eax
  _BYTE *v23; // rdx
  int v24; // edi

  v10 = sub_3181330(a2, a3, a4, a5);
  v11 = v10 ^ 1;
  LOBYTE(v11) = (a5 != 20) & (v10 ^ 1);
  if ( (_BYTE)v11 )
    return 0;
  sub_271D2D0((_BYTE *)a1);
  v12 = *(_BYTE *)(a1 + 2);
  if ( v12 != 1 )
  {
    if ( v12 > 1u && (unsigned __int8)(v12 - 2) > 1u )
      BUG();
    return v11;
  }
  sub_271D2E0(a1, 2);
  if ( !*(_BYTE *)(a1 + 100) )
    goto LABEL_15;
  v18 = *(_QWORD **)(a1 + 80);
  v15 = *(unsigned int *)(a1 + 92);
  v14 = &v18[v15];
  if ( v18 != v14 )
  {
    while ( a2 != (_BYTE *)*v18 )
    {
      if ( v14 == ++v18 )
        goto LABEL_14;
    }
    goto LABEL_16;
  }
LABEL_14:
  if ( (unsigned int)v15 < *(_DWORD *)(a1 + 88) )
  {
    *(_DWORD *)(a1 + 92) = v15 + 1;
    *v14 = a2;
    ++*(_QWORD *)(a1 + 72);
  }
  else
  {
LABEL_15:
    sub_C8CC70(a1 + 72, (__int64)a2, (__int64)v14, v15, v16, v17);
  }
LABEL_16:
  v11 = 1;
  if ( *a2 == 85 )
  {
    v19 = *(_DWORD *)(a6 + 24);
    v20 = *(_QWORD *)(a6 + 8);
    if ( v19 )
    {
      v21 = v19 - 1;
      v22 = (v19 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v23 = *(_BYTE **)(v20 + 16LL * v22);
      if ( a2 == v23 )
      {
LABEL_19:
        *(_BYTE *)(a1 + 120) = 1;
      }
      else
      {
        v24 = 1;
        while ( v23 != (_BYTE *)-4096LL )
        {
          v22 = v21 & (v24 + v22);
          v23 = *(_BYTE **)(v20 + 16LL * v22);
          if ( a2 == v23 )
            goto LABEL_19;
          ++v24;
        }
      }
      return 1;
    }
  }
  return v11;
}
