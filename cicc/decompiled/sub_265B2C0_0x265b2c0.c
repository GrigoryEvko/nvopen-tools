// Function: sub_265B2C0
// Address: 0x265b2c0
//
__int64 __fastcall sub_265B2C0(__int64 a1, __int64 a2, unsigned __int64 *a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v5; // rdx
  unsigned __int64 *v6; // rbx
  unsigned __int64 v7; // r14
  unsigned __int64 *v8; // rax
  _QWORD *v10; // rax
  char v11; // dl
  unsigned __int64 *v12; // r13
  _QWORD *i; // rax
  _QWORD *v14; // rdx
  bool v15; // r10
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  unsigned __int64 *v18; // rdx
  __int64 v19; // rax
  _QWORD *v20; // rax
  char v21; // [rsp+4h] [rbp-4Ch]
  _QWORD *v22; // [rsp+8h] [rbp-48h]
  _QWORD *v23; // [rsp+10h] [rbp-40h]

  if ( *(_QWORD *)(a2 + 120) )
  {
    v10 = sub_A19EB0((_QWORD *)(a2 + 80), a3);
    *(_BYTE *)(a1 + 8) = 0;
    *(_QWORD *)a1 = v10;
    *(_BYTE *)(a1 + 16) = v11;
  }
  else
  {
    v5 = *(unsigned int *)(a2 + 8);
    v6 = (unsigned __int64 *)(*(_QWORD *)a2 + 8 * v5);
    if ( *(unsigned __int64 **)a2 == v6 )
    {
      if ( v5 > 7 )
      {
        v23 = (_QWORD *)(a2 + 80);
LABEL_26:
        *(_DWORD *)(a2 + 8) = 0;
        v20 = sub_A19EB0(v23, a3);
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
      v8 = *(unsigned __int64 **)a2;
      while ( *v8 != v7 )
      {
        if ( v6 == ++v8 )
          goto LABEL_10;
      }
      if ( v6 != v8 )
      {
        *(_BYTE *)(a1 + 8) = 1;
        *(_QWORD *)a1 = v8;
        *(_BYTE *)(a1 + 16) = 0;
        return a1;
      }
LABEL_10:
      if ( v5 > 7 )
      {
        v12 = *(unsigned __int64 **)a2;
        v23 = (_QWORD *)(a2 + 80);
        for ( i = sub_265B1C0((_QWORD *)(a2 + 80), a2 + 88, *(unsigned __int64 **)a2); ; i = sub_265B1C0(
                                                                                               v23,
                                                                                               a2 + 88,
                                                                                               v12) )
        {
          if ( v14 )
          {
            v15 = i || v14 == (_QWORD *)(a2 + 88) || *v12 < v14[4];
            v21 = v15;
            v22 = v14;
            v16 = sub_22077B0(0x28u);
            *(_QWORD *)(v16 + 32) = *v12;
            sub_220F040(v21, v16, v22, (_QWORD *)(a2 + 88));
            ++*(_QWORD *)(a2 + 120);
          }
          if ( v6 == ++v12 )
            break;
        }
        goto LABEL_26;
      }
    }
    v17 = v5 + 1;
    if ( v17 > *(unsigned int *)(a2 + 12) )
    {
      sub_C8D5F0(a2, (const void *)(a2 + 16), v17, 8u, a5, *(_QWORD *)a2);
      v6 = (unsigned __int64 *)(*(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8));
    }
    *v6 = v7;
    v18 = *(unsigned __int64 **)a2;
    v19 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
    *(_DWORD *)(a2 + 8) = v19;
    *(_BYTE *)(a1 + 8) = 1;
    *(_QWORD *)a1 = &v18[v19 - 1];
    *(_BYTE *)(a1 + 16) = 1;
  }
  return a1;
}
