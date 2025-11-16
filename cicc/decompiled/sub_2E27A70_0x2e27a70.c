// Function: sub_2E27A70
// Address: 0x2e27a70
//
__int64 __fastcall sub_2E27A70(__int64 a1, __int64 a2, unsigned __int16 *a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v5; // rdx
  unsigned __int16 *v6; // rbx
  unsigned __int16 v7; // r14
  unsigned __int16 *v8; // rax
  __int64 v10; // rax
  char v11; // dl
  unsigned __int16 *v12; // r13
  __int64 i; // rax
  __int64 v14; // rdx
  bool v15; // r10
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  unsigned __int16 *v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  char v21; // [rsp+4h] [rbp-4Ch]
  _QWORD *v22; // [rsp+8h] [rbp-48h]
  _QWORD *v23; // [rsp+10h] [rbp-40h]

  if ( *(_QWORD *)(a2 + 80) )
  {
    v10 = sub_2E26F10(a2 + 40, a3);
    *(_BYTE *)(a1 + 8) = 0;
    *(_QWORD *)a1 = v10;
    *(_BYTE *)(a1 + 16) = v11;
  }
  else
  {
    v5 = *(_QWORD *)(a2 + 8);
    v6 = (unsigned __int16 *)(*(_QWORD *)a2 + 2 * v5);
    if ( *(unsigned __int16 **)a2 == v6 )
    {
      if ( v5 > 7 )
      {
        v23 = (_QWORD *)(a2 + 40);
LABEL_26:
        *(_QWORD *)(a2 + 8) = 0;
        v20 = sub_2E26F10((__int64)v23, a3);
        *(_BYTE *)(a1 + 8) = 0;
        *(_QWORD *)a1 = v20;
        *(_BYTE *)(a1 + 16) = 1;
        return a1;
      }
      v7 = *a3;
    }
    else
    {
      v7 = *a3;
      v8 = *(unsigned __int16 **)a2;
      while ( *v8 != v7 )
      {
        if ( v6 == ++v8 )
          goto LABEL_10;
      }
      if ( v8 != v6 )
      {
        *(_BYTE *)(a1 + 8) = 1;
        *(_QWORD *)a1 = v8;
        *(_BYTE *)(a1 + 16) = 0;
        return a1;
      }
LABEL_10:
      if ( v5 > 7 )
      {
        v12 = *(unsigned __int16 **)a2;
        v23 = (_QWORD *)(a2 + 40);
        for ( i = sub_2E27970((_QWORD *)(a2 + 40), a2 + 48, *(unsigned __int16 **)a2); ; i = sub_2E27970(
                                                                                               v23,
                                                                                               a2 + 48,
                                                                                               v12) )
        {
          if ( v14 )
          {
            v15 = i || v14 == a2 + 48 || *v12 < *(_WORD *)(v14 + 32);
            v21 = v15;
            v22 = (_QWORD *)v14;
            v16 = sub_22077B0(0x28u);
            *(_WORD *)(v16 + 32) = *v12;
            sub_220F040(v21, v16, v22, (_QWORD *)(a2 + 48));
            ++*(_QWORD *)(a2 + 80);
          }
          if ( v6 == ++v12 )
            break;
        }
        goto LABEL_26;
      }
    }
    v17 = v5 + 1;
    if ( v17 > *(_QWORD *)(a2 + 16) )
    {
      sub_C8D290(a2, (const void *)(a2 + 24), v17, 2u, a5, *(_QWORD *)a2);
      v6 = (unsigned __int16 *)(*(_QWORD *)a2 + 2LL * *(_QWORD *)(a2 + 8));
    }
    *v6 = v7;
    v18 = *(unsigned __int16 **)a2;
    v19 = *(_QWORD *)(a2 + 8) + 1LL;
    *(_QWORD *)(a2 + 8) = v19;
    *(_BYTE *)(a1 + 8) = 1;
    *(_QWORD *)a1 = &v18[v19 - 1];
    *(_BYTE *)(a1 + 16) = 1;
  }
  return a1;
}
