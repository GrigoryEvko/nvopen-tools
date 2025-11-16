// Function: sub_3516000
// Address: 0x3516000
//
__int64 __fastcall sub_3516000(__int64 a1, __int64 a2, _QWORD **a3, __int64 a4, __int64 a5)
{
  __int64 *v5; // r8
  unsigned int v8; // r12d
  __int64 *v9; // r14
  _QWORD *v11; // rdi
  _QWORD *v12; // rsi
  __int64 v13; // r8
  __int64 *v14; // r10
  _QWORD **v15; // rax
  __int64 v16; // r9
  __int64 v17; // r8
  bool v18; // zf
  unsigned int v19; // eax
  unsigned int v20; // edx
  bool v21; // cc
  int v23; // eax
  __int64 v24; // rcx
  int v25; // edx
  unsigned int v26; // eax
  __int64 v27; // rsi
  int v28; // edi
  __int64 v29; // rax
  __int64 v30; // [rsp+0h] [rbp-60h]
  __int64 *v33; // [rsp+18h] [rbp-48h]
  __int64 v34[7]; // [rsp+28h] [rbp-38h] BYREF

  v5 = *(__int64 **)(a2 + 112);
  v33 = &v5[*(unsigned int *)(a2 + 120)];
  if ( v5 != v33 )
  {
    v8 = 0x80000000;
    v9 = *(__int64 **)(a2 + 112);
    while ( 1 )
    {
      v13 = *v9;
      v18 = *(_BYTE *)(*v9 + 216) == 0;
      v34[0] = *v9;
      if ( !v18 )
        goto LABEL_10;
      if ( !a4 )
        break;
      if ( *(_DWORD *)(a4 + 16) )
      {
        v23 = *(_DWORD *)(a4 + 24);
        v24 = *(_QWORD *)(a4 + 8);
        if ( !v23 )
          goto LABEL_10;
        v25 = v23 - 1;
        v14 = v34;
        v26 = (v23 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v27 = *(_QWORD *)(v24 + 8LL * v26);
        if ( v13 != v27 )
        {
          v28 = 1;
          while ( v27 != -4096 )
          {
            v26 = v25 & (v28 + v26);
            v27 = *(_QWORD *)(v24 + 8LL * v26);
            if ( v13 == v27 )
              goto LABEL_20;
            ++v28;
          }
          goto LABEL_10;
        }
LABEL_6:
        v15 = (_QWORD **)*sub_3515040(a1 + 888, v14);
        if ( v15 == a3 )
        {
          v13 = v34[0];
          goto LABEL_10;
        }
        v17 = v34[0];
        if ( **v15 == v34[0] )
        {
          v29 = *(unsigned int *)(a5 + 8);
          if ( v29 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
          {
            v30 = v34[0];
            sub_C8D5F0(a5, (const void *)(a5 + 16), v29 + 1, 8u, v34[0], v16);
            v17 = v30;
            v29 = *(unsigned int *)(a5 + 8);
          }
          *(_QWORD *)(*(_QWORD *)a5 + 8 * v29) = v17;
          ++*(_DWORD *)(a5 + 8);
        }
        if ( v33 == ++v9 )
          return v8;
      }
      else
      {
        v11 = *(_QWORD **)(a4 + 32);
        v12 = &v11[*(unsigned int *)(a4 + 40)];
        if ( v12 != sub_3510810(v11, (__int64)v12, v34) )
          goto LABEL_6;
LABEL_10:
        v19 = sub_2E441D0(*(_QWORD *)(a1 + 528), a2, v13);
        v20 = v8 - v19;
        v21 = v19 <= v8;
        v8 = 0;
        if ( v21 )
          v8 = v20;
        if ( v33 == ++v9 )
          return v8;
      }
    }
LABEL_20:
    v14 = v34;
    goto LABEL_6;
  }
  return 0x80000000;
}
