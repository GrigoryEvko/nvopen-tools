// Function: sub_2872920
// Address: 0x2872920
//
__int64 *__fastcall sub_2872920(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r8
  unsigned __int64 v7; // rdx
  __int64 *v8; // r12
  __int64 v9; // r13
  __int64 *result; // rax
  char v11; // dl
  __int64 v12; // r12
  __int64 *v13; // r15
  _QWORD *i; // rax
  _QWORD *v15; // rdx
  bool v16; // r9
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  char v19; // [rsp+Ch] [rbp-44h]
  _QWORD *v20; // [rsp+10h] [rbp-40h]
  _QWORD *v21; // [rsp+18h] [rbp-38h]

  if ( *(_QWORD *)(a1 + 120) )
  {
    result = sub_25699C0(a1 + 80, a2);
    if ( !v11 )
      return result;
    goto LABEL_9;
  }
  v6 = *(__int64 **)a1;
  v7 = *(unsigned int *)(a1 + 8);
  v8 = (__int64 *)(*(_QWORD *)a1 + 8 * v7);
  if ( *(__int64 **)a1 != v8 )
  {
    v9 = *a2;
    result = *(__int64 **)a1;
    while ( *result != v9 )
    {
      if ( v8 == ++result )
        goto LABEL_12;
    }
    if ( v8 != result )
      return result;
LABEL_12:
    if ( v7 > 7 )
    {
      v13 = *(__int64 **)a1;
      v21 = (_QWORD *)(a1 + 80);
      for ( i = sub_FB52B0((_QWORD *)(a1 + 80), (_QWORD *)(a1 + 88), *(__int64 **)a1);
            ;
            i = sub_FB52B0(v21, (_QWORD *)(a1 + 88), v13) )
      {
        if ( v15 )
        {
          v16 = i || v15 == (_QWORD *)(a1 + 88) || *v13 < v15[4];
          v19 = v16;
          v20 = v15;
          v17 = sub_22077B0(0x28u);
          *(_QWORD *)(v17 + 32) = *v13;
          sub_220F040(v19, v17, v20, (_QWORD *)(a1 + 88));
          ++*(_QWORD *)(a1 + 120);
        }
        if ( v8 == ++v13 )
          break;
      }
      goto LABEL_23;
    }
LABEL_26:
    v18 = v7 + 1;
    if ( v18 > *(unsigned int *)(a1 + 12) )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), v18, 8u, (__int64)v6, a6);
      v8 = (__int64 *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8));
    }
    *v8 = v9;
    ++*(_DWORD *)(a1 + 8);
    goto LABEL_9;
  }
  if ( v7 <= 7 )
  {
    v9 = *a2;
    goto LABEL_26;
  }
  v21 = (_QWORD *)(a1 + 80);
LABEL_23:
  *(_DWORD *)(a1 + 8) = 0;
  sub_25699C0((__int64)v21, a2);
LABEL_9:
  result = (__int64 *)*(unsigned int *)(a1 + 136);
  v12 = *a2;
  if ( (unsigned __int64)result + 1 > *(unsigned int *)(a1 + 140) )
  {
    sub_C8D5F0(a1 + 128, (const void *)(a1 + 144), (unsigned __int64)result + 1, 8u, (__int64)v6, a6);
    result = (__int64 *)*(unsigned int *)(a1 + 136);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 128) + 8LL * (_QWORD)result) = v12;
  ++*(_DWORD *)(a1 + 136);
  return result;
}
