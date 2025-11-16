// Function: sub_2D301F0
// Address: 0x2d301f0
//
__int64 __fastcall sub_2D301F0(__int64 a1, _QWORD *a2, unsigned int a3, int a4)
{
  unsigned int v8; // esi
  __int64 v9; // rdi
  unsigned int v10; // eax
  int *v11; // rcx
  int v12; // edx
  __int64 result; // rax
  __int64 v14; // r11
  __int64 v15; // rdx
  unsigned int *v16; // r15
  unsigned int *v17; // rbx
  __int64 v18; // r8
  unsigned int *v19; // rdi
  unsigned int v20; // edx
  unsigned int v21; // ecx
  unsigned int v22; // esi
  unsigned int *v23; // r10
  int v24; // edx
  int v25; // eax
  int v26; // r10d
  int *v27; // r9
  int v28; // eax
  int v29; // edx
  int v30; // ecx
  int v31; // [rsp+14h] [rbp-4Ch]
  __int64 v32; // [rsp+18h] [rbp-48h]
  unsigned int v33; // [rsp+24h] [rbp-3Ch] BYREF
  _QWORD v34[7]; // [rsp+28h] [rbp-38h] BYREF

  *(_QWORD *)(*a2 + 8LL * (a3 >> 6)) |= 1LL << a3;
  *(_DWORD *)(a2[25] + 4LL * a3) = a4;
  v8 = *(_DWORD *)(a1 + 240);
  v33 = a3;
  v32 = a1 + 216;
  if ( !v8 )
  {
    ++*(_QWORD *)(a1 + 216);
    v34[0] = 0;
LABEL_35:
    v8 *= 2;
    goto LABEL_36;
  }
  v9 = *(_QWORD *)(a1 + 224);
  v10 = (v8 - 1) & (37 * a3);
  v11 = (int *)(v9 + 4LL * v10);
  v12 = *v11;
  if ( a3 == *v11 )
    goto LABEL_3;
  v26 = 1;
  v27 = 0;
  while ( v12 != -1 )
  {
    if ( v27 || v12 != -2 )
      v11 = v27;
    v10 = (v8 - 1) & (v26 + v10);
    v12 = *(_DWORD *)(v9 + 4LL * v10);
    if ( a3 == v12 )
      goto LABEL_3;
    ++v26;
    v27 = v11;
    v11 = (int *)(v9 + 4LL * v10);
  }
  v28 = *(_DWORD *)(a1 + 232);
  if ( !v27 )
    v27 = v11;
  ++*(_QWORD *)(a1 + 216);
  v29 = v28 + 1;
  v34[0] = v27;
  if ( 4 * (v28 + 1) >= 3 * v8 )
    goto LABEL_35;
  v30 = a3;
  if ( v8 - *(_DWORD *)(a1 + 236) - v29 <= v8 >> 3 )
  {
LABEL_36:
    sub_2D30060(v32, v8);
    sub_2D2B520(v32, (int *)&v33, v34);
    v30 = v33;
    v27 = (int *)v34[0];
    v29 = *(_DWORD *)(a1 + 232) + 1;
  }
  *(_DWORD *)(a1 + 232) = v29;
  if ( *v27 != -1 )
    --*(_DWORD *)(a1 + 236);
  *v27 = v30;
LABEL_3:
  result = sub_2D22AD0(a1, a3);
  v16 = (unsigned int *)(result + 4 * v15);
  v17 = (unsigned int *)result;
  if ( (unsigned int *)result != v16 )
  {
    while ( 1 )
    {
      v21 = *v17;
      *(_QWORD *)(*a2 + 8LL * (v21 >> 6)) |= v14 << v21;
      *(_DWORD *)(a2[25] + 4LL * v21) = a4;
      v22 = *(_DWORD *)(a1 + 240);
      v33 = v21;
      if ( !v22 )
        break;
      v18 = *(_QWORD *)(a1 + 224);
      result = (v22 - 1) & (37 * v21);
      v19 = (unsigned int *)(v18 + 4 * result);
      v20 = *v19;
      if ( v21 != *v19 )
      {
        v31 = 1;
        v23 = 0;
        while ( v20 != -1 )
        {
          if ( v23 || v20 != -2 )
            v19 = v23;
          result = (v22 - 1) & (v31 + (_DWORD)result);
          v20 = *(_DWORD *)(v18 + 4LL * (unsigned int)result);
          if ( v21 == v20 )
            goto LABEL_6;
          ++v31;
          v23 = v19;
          v19 = (unsigned int *)(v18 + 4LL * (unsigned int)result);
        }
        v25 = *(_DWORD *)(a1 + 232);
        if ( !v23 )
          v23 = v19;
        ++*(_QWORD *)(a1 + 216);
        v24 = v25 + 1;
        v34[0] = v23;
        if ( 4 * (v25 + 1) < 3 * v22 )
        {
          result = v22 - *(_DWORD *)(a1 + 236) - v24;
          if ( (unsigned int)result > v22 >> 3 )
            goto LABEL_18;
          goto LABEL_10;
        }
LABEL_9:
        v22 *= 2;
LABEL_10:
        sub_2D30060(v32, v22);
        sub_2D2B520(v32, (int *)&v33, v34);
        result = *(unsigned int *)(a1 + 232);
        v21 = v33;
        v14 = 1;
        v23 = (unsigned int *)v34[0];
        v24 = result + 1;
LABEL_18:
        *(_DWORD *)(a1 + 232) = v24;
        if ( *v23 != -1 )
          --*(_DWORD *)(a1 + 236);
        *v23 = v21;
      }
LABEL_6:
      if ( v16 == ++v17 )
        return result;
    }
    ++*(_QWORD *)(a1 + 216);
    v34[0] = 0;
    goto LABEL_9;
  }
  return result;
}
