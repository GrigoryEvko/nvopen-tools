// Function: sub_34C53D0
// Address: 0x34c53d0
//
unsigned __int64 __fastcall sub_34C53D0(unsigned int a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r8
  char *v8; // r9
  __int64 v9; // rdx
  unsigned __int64 result; // rax
  char *j; // r14
  unsigned int v12; // r15d
  bool v13; // zf
  unsigned __int64 v14; // rdx
  unsigned int *v15; // rbx
  unsigned int *v16; // r12
  bool v17; // r11
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  unsigned int *v21; // rdx
  unsigned __int64 v22; // rcx
  unsigned int *v23; // rbx
  unsigned __int64 v24; // rdx
  unsigned int *v25; // r14
  __int64 i; // rax
  __int64 v27; // rdx
  bool v28; // r9
  __int64 v29; // rax
  _QWORD *v30; // r15
  _QWORD *v31; // [rsp+10h] [rbp-60h]
  unsigned __int16 *v32; // [rsp+18h] [rbp-58h]
  char v33; // [rsp+18h] [rbp-58h]
  _QWORD *v34; // [rsp+20h] [rbp-50h]
  _QWORD *v35; // [rsp+20h] [rbp-50h]
  char v36; // [rsp+28h] [rbp-48h]
  unsigned int v37[4]; // [rsp+2Ch] [rbp-44h] BYREF
  unsigned int v38[13]; // [rsp+3Ch] [rbp-34h] BYREF

  v37[0] = a1;
  if ( a1 - 1 > 0x3FFFFFFE )
  {
    if ( *(_QWORD *)(a3 + 72) )
      return sub_2DCBF00(a3 + 32, v37);
    v21 = *(unsigned int **)a3;
    v22 = *(unsigned int *)(a3 + 8);
    v23 = &v21[v22];
    if ( v21 == v23 )
    {
      if ( v22 <= 3 )
      {
LABEL_30:
        result = *(unsigned int *)(a3 + 12);
        if ( v22 + 1 > result )
        {
          sub_C8D5F0(a3, (const void *)(a3 + 16), v22 + 1, 4u, a5, a6);
          result = *(_QWORD *)a3;
          v23 = (unsigned int *)(*(_QWORD *)a3 + 4LL * *(unsigned int *)(a3 + 8));
        }
        *v23 = a1;
        ++*(_DWORD *)(a3 + 8);
        return result;
      }
      v35 = (_QWORD *)(a3 + 32);
    }
    else
    {
      result = (unsigned __int64)v21;
      while ( a1 != *(_DWORD *)result )
      {
        result += 4LL;
        if ( v23 == (unsigned int *)result )
          goto LABEL_29;
      }
      if ( v23 != (unsigned int *)result )
        return result;
LABEL_29:
      if ( v22 <= 3 )
        goto LABEL_30;
      v25 = v21;
      v35 = (_QWORD *)(a3 + 32);
      for ( i = sub_2DCC990((_QWORD *)(a3 + 32), a3 + 40, v21); ; i = sub_2DCC990(v35, a3 + 40, v25) )
      {
        v30 = (_QWORD *)v27;
        if ( v27 )
        {
          v28 = i || a3 + 40 == v27 || *v25 < *(_DWORD *)(v27 + 32);
          v33 = v28;
          v29 = sub_22077B0(0x28u);
          *(_DWORD *)(v29 + 32) = *v25;
          sub_220F040(v33, v29, v30, (_QWORD *)(a3 + 40));
          ++*(_QWORD *)(a3 + 72);
        }
        if ( v23 == ++v25 )
          break;
      }
    }
    *(_DWORD *)(a3 + 8) = 0;
    return sub_2DCBF00((__int64)v35, v37);
  }
  v8 = sub_E922F0(a2, a1);
  v34 = (_QWORD *)(a3 + 32);
  result = (unsigned __int64)&v8[2 * v9];
  v32 = (unsigned __int16 *)result;
  for ( j = v8; v32 != (unsigned __int16 *)j; j += 2 )
  {
    v12 = *(unsigned __int16 *)j;
    v13 = *(_QWORD *)(a3 + 72) == 0;
    v38[0] = v12;
    if ( v13 )
    {
      v14 = *(unsigned int *)(a3 + 8);
      v15 = (unsigned int *)(*(_QWORD *)a3 + 4 * v14);
      if ( *(unsigned int **)a3 == v15 )
      {
        if ( v14 <= 3 )
          goto LABEL_33;
      }
      else
      {
        result = *(_QWORD *)a3;
        while ( v12 != *(_DWORD *)result )
        {
          result += 4LL;
          if ( v15 == (unsigned int *)result )
            goto LABEL_14;
        }
        if ( v15 != (unsigned int *)result )
          continue;
LABEL_14:
        if ( v14 <= 3 )
        {
LABEL_33:
          result = *(unsigned int *)(a3 + 12);
          v24 = v14 + 1;
          if ( v24 > result )
          {
            sub_C8D5F0(a3, (const void *)(a3 + 16), v24, 4u, v7, (__int64)v8);
            result = *(_QWORD *)a3;
            v15 = (unsigned int *)(*(_QWORD *)a3 + 4LL * *(unsigned int *)(a3 + 8));
          }
          *v15 = v12;
          ++*(_DWORD *)(a3 + 8);
          continue;
        }
        v16 = *(unsigned int **)a3;
        do
        {
          v19 = sub_2DCC990(v34, a3 + 40, v16);
          if ( v20 )
          {
            v17 = v19 || v20 == a3 + 40 || *v16 < *(_DWORD *)(v20 + 32);
            v36 = v17;
            v31 = (_QWORD *)v20;
            v18 = sub_22077B0(0x28u);
            *(_DWORD *)(v18 + 32) = *v16;
            sub_220F040(v36, v18, v31, (_QWORD *)(a3 + 40));
            ++*(_QWORD *)(a3 + 72);
          }
          ++v16;
        }
        while ( v15 != v16 );
      }
      *(_DWORD *)(a3 + 8) = 0;
    }
    result = sub_2DCBE50((__int64)v34, v38);
  }
  return result;
}
