// Function: sub_2E02840
// Address: 0x2e02840
//
__int64 __fastcall sub_2E02840(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rdx
  __int64 *v7; // rbx
  __int64 v8; // r14
  __int64 *v9; // rax
  _QWORD *v11; // rax
  char v12; // dl
  __int64 *v13; // r13
  _QWORD *i; // rax
  _QWORD *v15; // rdx
  char v16; // r10
  __int64 v17; // rax
  bool v18; // al
  _QWORD *v19; // rax
  unsigned __int64 v20; // rdx
  __int64 *v21; // rdx
  __int64 v22; // rax
  _QWORD *v23; // [rsp+0h] [rbp-50h]
  char v24; // [rsp+8h] [rbp-48h]
  _QWORD *v25; // [rsp+8h] [rbp-48h]
  _QWORD *v26; // [rsp+10h] [rbp-40h]

  if ( *(_QWORD *)(a2 + 72) )
  {
    v11 = sub_2E00930(a2 + 32, a3);
    *(_BYTE *)(a1 + 8) = 0;
    *(_QWORD *)a1 = v11;
    *(_BYTE *)(a1 + 16) = v12;
  }
  else
  {
    v6 = *(unsigned int *)(a2 + 8);
    v7 = (__int64 *)(*(_QWORD *)a2 + 8 * v6);
    if ( *(__int64 **)a2 == v7 )
    {
      if ( v6 > 1 )
      {
        v26 = (_QWORD *)(a2 + 32);
LABEL_21:
        *(_DWORD *)(a2 + 8) = 0;
        v19 = sub_2E00930((__int64)v26, a3);
        *(_BYTE *)(a1 + 8) = 0;
        *(_QWORD *)a1 = v19;
        *(_BYTE *)(a1 + 16) = 1;
        return a1;
      }
      v8 = *a3;
    }
    else
    {
      v8 = *a3;
      v9 = *(__int64 **)a2;
      while ( v8 != *v9 )
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
        v13 = *(__int64 **)a2;
        v26 = (_QWORD *)(a2 + 32);
        for ( i = sub_2E026D0((_QWORD *)(a2 + 32), a2 + 40, *(__int64 **)a2); ; i = sub_2E026D0(v26, a2 + 40, v13) )
        {
          if ( v15 )
          {
            if ( i || v15 == (_QWORD *)(a2 + 40) )
            {
              v16 = 1;
            }
            else
            {
              v25 = v15;
              v18 = sub_2DF8300(v13, v15[4]);
              v15 = v25;
              v16 = v18;
            }
            v23 = v15;
            v24 = v16;
            v17 = sub_22077B0(0x28u);
            *(_QWORD *)(v17 + 32) = *v13;
            sub_220F040(v24, v17, v23, (_QWORD *)(a2 + 40));
            ++*(_QWORD *)(a2 + 72);
          }
          if ( v7 == ++v13 )
            break;
        }
        goto LABEL_21;
      }
    }
    v20 = v6 + 1;
    if ( v20 > *(unsigned int *)(a2 + 12) )
    {
      sub_C8D5F0(a2, (const void *)(a2 + 16), v20, 8u, *(_QWORD *)a2, a6);
      v7 = (__int64 *)(*(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8));
    }
    *v7 = v8;
    v21 = *(__int64 **)a2;
    v22 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
    *(_DWORD *)(a2 + 8) = v22;
    *(_BYTE *)(a1 + 8) = 1;
    *(_QWORD *)a1 = &v21[v22 - 1];
    *(_BYTE *)(a1 + 16) = 1;
  }
  return a1;
}
