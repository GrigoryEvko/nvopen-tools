// Function: sub_3228E60
// Address: 0x3228e60
//
__int64 __fastcall sub_3228E60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  const void *v7; // rax
  __int64 v8; // rdi
  char v9; // dl
  __int64 *v10; // rax
  unsigned int *v11; // r13
  unsigned __int64 v12; // rcx
  unsigned int *v13; // r15
  unsigned int v14; // r14d
  unsigned int *v15; // rax
  __int64 result; // rax
  __int64 v17; // rax
  unsigned int *v18; // r12
  bool v19; // r11
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  const void *v23; // [rsp+0h] [rbp-60h]
  _QWORD *v24; // [rsp+8h] [rbp-58h]
  char v25; // [rsp+14h] [rbp-4Ch]
  char v26; // [rsp+14h] [rbp-4Ch]
  _QWORD *v28; // [rsp+20h] [rbp-40h]

  v28 = (_QWORD *)(a1 + 80);
  v7 = (const void *)(a1 + 16);
  v8 = *(_QWORD *)a2;
  v23 = v7;
  v9 = *(_BYTE *)(a2 + 8);
LABEL_2:
  v10 = (__int64 *)a3;
  if ( *(_BYTE *)(a3 + 8) == v9 )
    goto LABEL_12;
LABEL_3:
  v11 = (unsigned int *)v8;
  if ( !v9 )
  {
LABEL_14:
    v11 = (unsigned int *)(v8 + 32);
    if ( *(_QWORD *)(a1 + 120) )
      goto LABEL_15;
    goto LABEL_5;
  }
  while ( 1 )
  {
    if ( *(_QWORD *)(a1 + 120) )
    {
LABEL_15:
      sub_2DCBF00((__int64)v28, v11);
      v9 = *(_BYTE *)(a2 + 8);
      v8 = *(_QWORD *)a2;
      if ( !v9 )
        goto LABEL_16;
      goto LABEL_11;
    }
LABEL_5:
    v12 = *(unsigned int *)(a1 + 8);
    v13 = (unsigned int *)(*(_QWORD *)a1 + 4 * v12);
    if ( *(unsigned int **)a1 == v13 )
    {
      if ( v12 > 0xF )
        goto LABEL_29;
      v14 = *v11;
    }
    else
    {
      v14 = *v11;
      v15 = *(unsigned int **)a1;
      while ( *v15 != v14 )
      {
        if ( v13 == ++v15 )
          goto LABEL_19;
      }
      if ( v13 != v15 )
        goto LABEL_10;
LABEL_19:
      if ( v12 > 0xF )
      {
        v18 = *(unsigned int **)a1;
        do
        {
          v21 = sub_2DCC990(v28, a1 + 88, v18);
          if ( v22 )
          {
            v19 = v21 || v22 == a1 + 88 || *v18 < *(_DWORD *)(v22 + 32);
            v24 = (_QWORD *)v22;
            v26 = v19;
            v20 = sub_22077B0(0x28u);
            *(_DWORD *)(v20 + 32) = *v18;
            sub_220F040(v26, v20, v24, (_QWORD *)(a1 + 88));
            ++*(_QWORD *)(a1 + 120);
          }
          ++v18;
        }
        while ( v13 != v18 );
LABEL_29:
        *(_DWORD *)(a1 + 8) = 0;
        goto LABEL_15;
      }
    }
    if ( v12 + 1 > *(unsigned int *)(a1 + 12) )
    {
      sub_C8D5F0(a1, v23, v12 + 1, 4u, a5, a6);
      v13 = (unsigned int *)(*(_QWORD *)a1 + 4LL * *(unsigned int *)(a1 + 8));
    }
    *v13 = v14;
    ++*(_DWORD *)(a1 + 8);
    v9 = *(_BYTE *)(a2 + 8);
    v8 = *(_QWORD *)a2;
LABEL_10:
    if ( !v9 )
    {
LABEL_16:
      v25 = v9;
      v17 = sub_220EF30(v8);
      v9 = v25;
      v8 = v17;
      *(_QWORD *)a2 = v17;
      goto LABEL_2;
    }
LABEL_11:
    v8 += 4;
    *(_QWORD *)a2 = v8;
    v10 = (__int64 *)a3;
    if ( *(_BYTE *)(a3 + 8) != v9 )
      goto LABEL_3;
LABEL_12:
    result = *v10;
    if ( !v9 )
    {
      if ( v8 == result )
        return result;
      goto LABEL_14;
    }
    if ( v8 == result )
      return result;
    v11 = (unsigned int *)v8;
  }
}
