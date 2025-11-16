// Function: sub_372C4F0
// Address: 0x372c4f0
//
__int64 __fastcall sub_372C4F0(__int64 a1, __int64 a2, unsigned int *a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v5; // rdx
  unsigned int *v6; // rbx
  unsigned int v7; // r14d
  unsigned int *v8; // rax
  __int64 v10; // rax
  char v11; // dl
  unsigned int *v12; // r13
  __int64 i; // rax
  __int64 v14; // rdx
  bool v15; // r10
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  unsigned int *v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  _QWORD *v21; // [rsp+0h] [rbp-50h]
  char v22; // [rsp+Ch] [rbp-44h]
  _QWORD *v23; // [rsp+10h] [rbp-40h]

  if ( *(_QWORD *)(a2 + 72) )
  {
    v10 = sub_2DCBE50(a2 + 32, a3);
    *(_BYTE *)(a1 + 8) = 0;
    *(_QWORD *)a1 = v10;
    *(_BYTE *)(a1 + 16) = v11;
  }
  else
  {
    v5 = *(unsigned int *)(a2 + 8);
    v6 = (unsigned int *)(*(_QWORD *)a2 + 4 * v5);
    if ( *(unsigned int **)a2 == v6 )
    {
      if ( v5 > 3 )
      {
        v23 = (_QWORD *)(a2 + 32);
LABEL_26:
        *(_DWORD *)(a2 + 8) = 0;
        v20 = sub_2DCBE50((__int64)v23, a3);
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
      v8 = *(unsigned int **)a2;
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
      if ( v5 > 3 )
      {
        v12 = *(unsigned int **)a2;
        v23 = (_QWORD *)(a2 + 32);
        for ( i = sub_2DCC990((_QWORD *)(a2 + 32), a2 + 40, *(unsigned int **)a2); ; i = sub_2DCC990(v23, a2 + 40, v12) )
        {
          if ( v14 )
          {
            v15 = i || v14 == a2 + 40 || *v12 < *(_DWORD *)(v14 + 32);
            v21 = (_QWORD *)v14;
            v22 = v15;
            v16 = sub_22077B0(0x28u);
            *(_DWORD *)(v16 + 32) = *v12;
            sub_220F040(v22, v16, v21, (_QWORD *)(a2 + 40));
            ++*(_QWORD *)(a2 + 72);
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
      sub_C8D5F0(a2, (const void *)(a2 + 16), v17, 4u, a5, *(_QWORD *)a2);
      v6 = (unsigned int *)(*(_QWORD *)a2 + 4LL * *(unsigned int *)(a2 + 8));
    }
    *v6 = v7;
    v18 = *(unsigned int **)a2;
    v19 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
    *(_DWORD *)(a2 + 8) = v19;
    *(_BYTE *)(a1 + 8) = 1;
    *(_QWORD *)a1 = &v18[v19 - 1];
    *(_BYTE *)(a1 + 16) = 1;
  }
  return a1;
}
