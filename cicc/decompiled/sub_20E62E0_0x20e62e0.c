// Function: sub_20E62E0
// Address: 0x20e62e0
//
__int64 __fastcall sub_20E62E0(_QWORD *a1, __int64 a2, __int64 a3, unsigned int a4, int a5, __int64 **a6, __int64 a7)
{
  __int64 v9; // rdi
  __int64 v10; // rbx
  int v11; // ebx
  __int64 v12; // rax
  unsigned int v13; // r12d
  __int64 v14; // rdx
  int *v15; // r8
  __int64 v16; // rdi
  __int64 v17; // r11
  int v18; // esi
  __int64 v19; // rdx
  __int64 v20; // r9
  unsigned int v21; // ecx
  _WORD *v22; // r10
  unsigned __int16 v23; // ax
  __int16 *v24; // rcx
  _WORD *v25; // r10
  int v26; // edx
  unsigned __int16 *v27; // r9
  unsigned int v28; // r10d
  unsigned int i; // esi
  __int16 *v30; // r10
  __int16 v31; // cx
  int v32; // esi
  __int64 v35; // [rsp+18h] [rbp-48h]
  __int64 v38; // [rsp+28h] [rbp-38h]

  v9 = a1[5];
  v10 = *(_QWORD *)v9 + 24LL * *((unsigned __int16 *)*a6 + 12);
  if ( *(_DWORD *)(v9 + 8) != *(_DWORD *)v10 )
    sub_1ED7890(v9, a6);
  v35 = *(unsigned int *)(v10 + 4);
  v38 = *(_QWORD *)(v10 + 16);
  if ( *(_DWORD *)(v10 + 4) )
  {
    v11 = 0;
    v12 = 0;
    do
    {
      v13 = *(unsigned __int16 *)(v38 + 2 * v12);
      if ( v13 != a5
        && v13 != a4
        && !(unsigned __int8)sub_20E61E0((__int64)a1, a2, a3, *(unsigned __int16 *)(v38 + 2 * v12)) )
      {
        v14 = a1[18];
        if ( *(_DWORD *)(v14 + 4LL * (unsigned __int16)v13) == -1
          && *(_QWORD *)(a1[9] + 8LL * (unsigned __int16)v13) != -1
          && *(_DWORD *)(v14 + 4LL * a4) <= *(_DWORD *)(a1[21] + 4LL * (unsigned __int16)v13) )
        {
          v15 = *(int **)a7;
          v16 = *(_QWORD *)a7 + 4LL * *(unsigned int *)(a7 + 8);
          if ( *(_QWORD *)a7 == v16 )
            return v13;
          v17 = a1[4];
          while ( 1 )
          {
            v18 = *v15;
            if ( *v15 == v13 )
              break;
            if ( v18 >= 0 )
            {
              v19 = *(_QWORD *)(v17 + 8);
              v20 = *(_QWORD *)(v17 + 56);
              v21 = *(_DWORD *)(v19 + 24LL * (unsigned __int16)v13 + 16);
              v22 = (_WORD *)(v20 + 2LL * (v21 >> 4));
              v23 = *v22 + v13 * (v21 & 0xF);
              v24 = v22 + 1;
              LODWORD(v22) = *(_DWORD *)(v19 + 24LL * (unsigned int)v18 + 16);
              v26 = v18 * ((unsigned __int8)v22 & 0xF);
              v25 = (_WORD *)(v20 + 2LL * ((unsigned int)v22 >> 4));
              LOWORD(v26) = *v25 + v26;
              v27 = v25 + 1;
              v28 = v23;
              for ( i = (unsigned __int16)v26; v28 != i; i = (unsigned __int16)v26 )
              {
                if ( v28 < i )
                {
                  do
                  {
                    v30 = v24 + 1;
                    v31 = *v24;
                    v23 += v31;
                    if ( !v31 )
                      goto LABEL_24;
                    v24 = v30;
                    v28 = v23;
                    if ( v23 == i )
                      goto LABEL_5;
                  }
                  while ( v23 < i );
                }
                v32 = *v27;
                if ( !(_WORD)v32 )
                  goto LABEL_24;
                v26 += v32;
                ++v27;
              }
              break;
            }
LABEL_24:
            if ( (int *)v16 == ++v15 )
              return v13;
          }
        }
      }
LABEL_5:
      v12 = (unsigned int)++v11;
    }
    while ( v11 != v35 );
  }
  return 0;
}
