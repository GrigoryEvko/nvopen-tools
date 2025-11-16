// Function: sub_14109A0
// Address: 0x14109a0
//
__int64 __fastcall sub_14109A0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rax
  _QWORD *v9; // rdi
  _QWORD *v10; // rcx
  __int64 v11; // rdx
  unsigned int v12; // eax
  __int64 v13; // rdx
  __int64 v15; // rax
  _QWORD *v16; // r14
  _QWORD *v17; // rax
  __int64 v18; // rdx
  _QWORD *i; // rbx
  __int64 v20; // rax
  __int64 v21; // rdi
  unsigned int v22; // esi
  __int64 *v23; // r13
  int v24; // ecx
  __int64 v25; // r9
  __int64 v26; // rax
  __int64 v27; // rdx
  _QWORD *v28; // rdi
  _QWORD *v29; // rax
  __int64 v30; // [rsp+8h] [rbp-38h]

  v7 = sub_15A9650(*(_QWORD *)a1, *a2, a3, a4, a5, a6);
  *(_QWORD *)(a1 + 96) = v7;
  *(_QWORD *)(a1 + 104) = sub_159C470(v7, 0, 0);
  v8 = sub_1410110(a1, a2);
  v9 = *(_QWORD **)(a1 + 160);
  v10 = *(_QWORD **)(a1 + 152);
  v30 = v8;
  if ( !v11 || !v8 )
  {
    v15 = v9 == v10 ? *(unsigned int *)(a1 + 172) : *(unsigned int *)(a1 + 168);
    v16 = &v9[v15];
    if ( v9 != v16 )
    {
      v17 = v9;
      v18 = *v9;
      for ( i = v9; *v17 >= 0xFFFFFFFFFFFFFFFELL; i = v17 )
      {
        if ( v16 == ++v17 )
          goto LABEL_3;
        v18 = *v17;
      }
      if ( v16 != v17 )
      {
        while ( 1 )
        {
          v20 = *(unsigned int *)(a1 + 136);
          if ( (_DWORD)v20 )
          {
            v21 = *(_QWORD *)(a1 + 120);
            v22 = (v20 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
            v23 = (__int64 *)(v21 + 56LL * v22);
            v24 = 1;
            v25 = *v23;
            if ( v18 != *v23 )
            {
              while ( v25 != -8 )
              {
                v22 = (v20 - 1) & (v24 + v22);
                v23 = (__int64 *)(v21 + 56LL * v22);
                v25 = *v23;
                if ( *v23 == v18 )
                  goto LABEL_19;
                ++v24;
              }
              goto LABEL_29;
            }
LABEL_19:
            if ( v23 != (__int64 *)(v21 + 56 * v20) )
            {
              v26 = v23[3];
              v27 = v23[6];
              if ( v26 )
              {
                if ( v27 )
                {
                  v28 = v23 + 4;
                  if ( v27 != -16 && v27 != -8 )
                    goto LABEL_24;
                }
                goto LABEL_25;
              }
              if ( v27 )
              {
                v28 = v23 + 4;
                if ( v27 == -8 || v27 == -16 )
                {
LABEL_28:
                  *v23 = -16;
                  --*(_DWORD *)(a1 + 128);
                  ++*(_DWORD *)(a1 + 132);
                  goto LABEL_29;
                }
LABEL_24:
                sub_1649B30(v28);
                v26 = v23[3];
LABEL_25:
                if ( v26 != 0 && v26 != -8 && v26 != -16 )
                  sub_1649B30(v23 + 1);
                goto LABEL_28;
              }
            }
          }
LABEL_29:
          v29 = i + 1;
          if ( i + 1 != v16 )
          {
            while ( 1 )
            {
              v18 = *v29;
              i = v29;
              if ( *v29 < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( v16 == ++v29 )
                goto LABEL_32;
            }
            if ( v16 != v29 )
              continue;
          }
LABEL_32:
          v9 = *(_QWORD **)(a1 + 160);
          v10 = *(_QWORD **)(a1 + 152);
          break;
        }
      }
    }
  }
LABEL_3:
  ++*(_QWORD *)(a1 + 144);
  if ( v10 != v9 )
  {
    v12 = 4 * (*(_DWORD *)(a1 + 172) - *(_DWORD *)(a1 + 176));
    v13 = *(unsigned int *)(a1 + 168);
    if ( v12 < 0x20 )
      v12 = 32;
    if ( (unsigned int)v13 > v12 )
    {
      sub_16CC920(a1 + 144);
      return v30;
    }
    memset(v9, -1, 8 * v13);
  }
  *(_QWORD *)(a1 + 172) = 0;
  return v30;
}
