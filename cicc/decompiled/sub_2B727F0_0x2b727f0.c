// Function: sub_2B727F0
// Address: 0x2b727f0
//
__int64 __fastcall sub_2B727F0(__int64 a1, __int64 a2, unsigned __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rdx
  unsigned __int64 *v7; // rbx
  unsigned __int64 v8; // r14
  unsigned __int64 *v9; // rax
  _QWORD *v11; // rax
  char v12; // dl
  unsigned __int64 *v13; // r13
  _QWORD *i; // rax
  _QWORD *v15; // rdx
  bool v16; // r10
  __int64 v17; // rax
  _QWORD *v18; // rax
  unsigned __int64 v19; // rdx
  unsigned __int64 *v20; // rdx
  __int64 v21; // rax
  char v22; // [rsp+4h] [rbp-4Ch]
  _QWORD *v23; // [rsp+8h] [rbp-48h]
  _QWORD *v24; // [rsp+10h] [rbp-40h]

  if ( *(_QWORD *)(a2 + 72) )
  {
    v11 = sub_A19EB0((_QWORD *)(a2 + 32), a3);
    *(_BYTE *)(a1 + 8) = 0;
    *(_QWORD *)a1 = v11;
    *(_BYTE *)(a1 + 16) = v12;
  }
  else
  {
    v6 = *(unsigned int *)(a2 + 8);
    v7 = (unsigned __int64 *)(*(_QWORD *)a2 + 8 * v6);
    if ( *(unsigned __int64 **)a2 == v7 )
    {
      if ( v6 > 1 )
      {
        v24 = (_QWORD *)(a2 + 32);
LABEL_21:
        *(_DWORD *)(a2 + 8) = 0;
        v18 = sub_A19EB0(v24, a3);
        *(_BYTE *)(a1 + 8) = 0;
        *(_QWORD *)a1 = v18;
        *(_BYTE *)(a1 + 16) = 1;
        return a1;
      }
      v8 = *a3;
    }
    else
    {
      v8 = *a3;
      v9 = *(unsigned __int64 **)a2;
      while ( *v9 != v8 )
      {
        if ( v7 == ++v9 )
          goto LABEL_10;
      }
      if ( v7 != v9 )
      {
        *(_BYTE *)(a1 + 8) = 1;
        *(_QWORD *)a1 = v9;
        *(_BYTE *)(a1 + 16) = 0;
        return a1;
      }
LABEL_10:
      if ( v6 > 1 )
      {
        v13 = *(unsigned __int64 **)a2;
        v24 = (_QWORD *)(a2 + 32);
        for ( i = sub_265B1C0((_QWORD *)(a2 + 32), a2 + 40, *(unsigned __int64 **)a2); ; i = sub_265B1C0(
                                                                                               v24,
                                                                                               a2 + 40,
                                                                                               v13) )
        {
          if ( v15 )
          {
            v16 = i || v15 == (_QWORD *)(a2 + 40) || *v13 < v15[4];
            v22 = v16;
            v23 = v15;
            v17 = sub_22077B0(0x28u);
            *(_QWORD *)(v17 + 32) = *v13;
            sub_220F040(v22, v17, v23, (_QWORD *)(a2 + 40));
            ++*(_QWORD *)(a2 + 72);
          }
          if ( v7 == ++v13 )
            break;
        }
        goto LABEL_21;
      }
    }
    v19 = v6 + 1;
    if ( v19 > *(unsigned int *)(a2 + 12) )
    {
      sub_C8D5F0(a2, (const void *)(a2 + 16), v19, 8u, *(_QWORD *)a2, a6);
      v7 = (unsigned __int64 *)(*(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8));
    }
    *v7 = v8;
    v20 = *(unsigned __int64 **)a2;
    v21 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
    *(_DWORD *)(a2 + 8) = v21;
    *(_BYTE *)(a1 + 8) = 1;
    *(_QWORD *)a1 = &v20[v21 - 1];
    *(_BYTE *)(a1 + 16) = 1;
  }
  return a1;
}
