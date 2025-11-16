// Function: sub_3536E40
// Address: 0x3536e40
//
__int64 __fastcall sub_3536E40(__int64 a1, __int64 a2, unsigned int *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rdx
  unsigned int *v7; // rbx
  unsigned int v8; // r14d
  unsigned int *v9; // rax
  __int64 v11; // rax
  char v12; // dl
  unsigned int *v13; // r13
  __int64 i; // rax
  __int64 v15; // rdx
  bool v16; // r10
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  unsigned int *v20; // rdx
  __int64 v21; // rax
  _QWORD *v22; // [rsp+0h] [rbp-50h]
  char v23; // [rsp+Ch] [rbp-44h]
  _QWORD *v24; // [rsp+10h] [rbp-40h]

  if ( *(_QWORD *)(a2 + 64) )
  {
    v11 = sub_2DCBE50(a2 + 24, a3);
    *(_BYTE *)(a1 + 8) = 0;
    *(_QWORD *)a1 = v11;
    *(_BYTE *)(a1 + 16) = v12;
  }
  else
  {
    v6 = *(unsigned int *)(a2 + 8);
    v7 = (unsigned int *)(*(_QWORD *)a2 + 4 * v6);
    if ( *(unsigned int **)a2 == v7 )
    {
      if ( v6 > 1 )
      {
        v24 = (_QWORD *)(a2 + 24);
LABEL_21:
        *(_DWORD *)(a2 + 8) = 0;
        v18 = sub_2DCBE50((__int64)v24, a3);
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
      v9 = *(unsigned int **)a2;
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
        v13 = *(unsigned int **)a2;
        v24 = (_QWORD *)(a2 + 24);
        for ( i = sub_2DCC990((_QWORD *)(a2 + 24), a2 + 32, *(unsigned int **)a2); ; i = sub_2DCC990(v24, a2 + 32, v13) )
        {
          if ( v15 )
          {
            v16 = i || v15 == a2 + 32 || *v13 < *(_DWORD *)(v15 + 32);
            v22 = (_QWORD *)v15;
            v23 = v16;
            v17 = sub_22077B0(0x28u);
            *(_DWORD *)(v17 + 32) = *v13;
            sub_220F040(v23, v17, v22, (_QWORD *)(a2 + 32));
            ++*(_QWORD *)(a2 + 64);
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
      sub_C8D5F0(a2, (const void *)(a2 + 16), v19, 4u, *(_QWORD *)a2, a6);
      v7 = (unsigned int *)(*(_QWORD *)a2 + 4LL * *(unsigned int *)(a2 + 8));
    }
    *v7 = v8;
    v20 = *(unsigned int **)a2;
    v21 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
    *(_DWORD *)(a2 + 8) = v21;
    *(_BYTE *)(a1 + 8) = 1;
    *(_QWORD *)a1 = &v20[v21 - 1];
    *(_BYTE *)(a1 + 16) = 1;
  }
  return a1;
}
