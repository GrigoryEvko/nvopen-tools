// Function: sub_372C6D0
// Address: 0x372c6d0
//
__int64 __fastcall sub_372C6D0(__int64 a1, __int64 a2, unsigned __int64 *a3, __int64 a4, __int64 a5)
{
  unsigned __int64 *v5; // rbx
  unsigned __int64 v6; // r14
  unsigned __int64 *v7; // rax
  _QWORD *v9; // rax
  char v10; // dl
  unsigned __int64 *v11; // r13
  _QWORD *i; // rax
  _QWORD *v13; // rdx
  bool v14; // r10
  __int64 v15; // rax
  unsigned __int64 *v16; // rdx
  __int64 v17; // rax
  _QWORD *v18; // rax
  char v19; // [rsp+4h] [rbp-4Ch]
  _QWORD *v20; // [rsp+8h] [rbp-48h]
  _QWORD *v21; // [rsp+10h] [rbp-40h]

  if ( *(_QWORD *)(a2 + 64) )
  {
    v9 = sub_A19EB0((_QWORD *)(a2 + 24), a3);
    *(_BYTE *)(a1 + 8) = 0;
    *(_QWORD *)a1 = v9;
    *(_BYTE *)(a1 + 16) = v10;
  }
  else
  {
    v5 = (unsigned __int64 *)(*(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8));
    if ( *(unsigned __int64 **)a2 == v5 )
    {
      if ( *(_DWORD *)(a2 + 8) )
      {
        v21 = (_QWORD *)(a2 + 24);
LABEL_26:
        *(_DWORD *)(a2 + 8) = 0;
        v18 = sub_A19EB0(v21, a3);
        *(_BYTE *)(a1 + 8) = 0;
        *(_QWORD *)a1 = v18;
        *(_BYTE *)(a1 + 16) = 1;
        return a1;
      }
      v6 = *a3;
    }
    else
    {
      v6 = *a3;
      v7 = *(unsigned __int64 **)a2;
      while ( *v7 != v6 )
      {
        if ( v5 == ++v7 )
          goto LABEL_10;
      }
      if ( v5 != v7 )
      {
        *(_BYTE *)(a1 + 8) = 1;
        *(_QWORD *)a1 = v7;
        *(_BYTE *)(a1 + 16) = 0;
        return a1;
      }
LABEL_10:
      if ( *(_DWORD *)(a2 + 8) )
      {
        v11 = *(unsigned __int64 **)a2;
        v21 = (_QWORD *)(a2 + 24);
        for ( i = sub_265B1C0((_QWORD *)(a2 + 24), a2 + 32, *(unsigned __int64 **)a2); ; i = sub_265B1C0(
                                                                                               v21,
                                                                                               a2 + 32,
                                                                                               v11) )
        {
          if ( v13 )
          {
            v14 = i || v13 == (_QWORD *)(a2 + 32) || *v11 < v13[4];
            v19 = v14;
            v20 = v13;
            v15 = sub_22077B0(0x28u);
            *(_QWORD *)(v15 + 32) = *v11;
            sub_220F040(v19, v15, v20, (_QWORD *)(a2 + 32));
            ++*(_QWORD *)(a2 + 64);
          }
          if ( v5 == ++v11 )
            break;
        }
        goto LABEL_26;
      }
    }
    if ( !*(_DWORD *)(a2 + 12) )
    {
      sub_C8D5F0(a2, (const void *)(a2 + 16), 1u, 8u, a5, *(_QWORD *)a2);
      v5 = (unsigned __int64 *)(*(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8));
    }
    *v5 = v6;
    v16 = *(unsigned __int64 **)a2;
    v17 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
    *(_DWORD *)(a2 + 8) = v17;
    *(_BYTE *)(a1 + 8) = 1;
    *(_QWORD *)a1 = &v16[v17 - 1];
    *(_BYTE *)(a1 + 16) = 1;
  }
  return a1;
}
