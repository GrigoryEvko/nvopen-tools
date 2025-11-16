// Function: sub_2180DE0
// Address: 0x2180de0
//
__int64 __fastcall sub_2180DE0(
        __int64 a1,
        __int64 a2,
        int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        unsigned int a8)
{
  int v8; // r13d
  unsigned int v11; // r15d
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rbx
  __int64 v16; // r15
  __int64 v17; // r13
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // r12
  __int64 v21; // rdx
  __int64 v22; // rcx
  int v23; // r8d
  int v24; // r9d
  __int64 v25; // rax
  __int64 v26; // rax
  int v27; // r8d
  int v28; // r9d
  unsigned __int64 v29; // rcx
  _DWORD *v30; // r12
  __int64 v31; // rcx
  int v32; // edx
  _DWORD *v33; // rbx
  _DWORD *v34; // r12
  _DWORD *v35; // rsi
  unsigned int v36; // r12d
  __int64 v37; // rsi
  __int64 v38; // rax
  _DWORD *v39; // rdx
  _DWORD *i; // rax
  __int64 v41; // [rsp+8h] [rbp-A8h]
  int v42; // [rsp+10h] [rbp-A0h]
  unsigned __int8 v44; // [rsp+33h] [rbp-7Dh]
  int v45; // [rsp+34h] [rbp-7Ch] BYREF
  __int64 v46; // [rsp+38h] [rbp-78h] BYREF
  int v47; // [rsp+4Ch] [rbp-64h] BYREF
  _QWORD v48[4]; // [rsp+50h] [rbp-60h] BYREF
  unsigned __int8 v49; // [rsp+70h] [rbp-40h]

  v8 = a5;
  v46 = a2;
  v45 = a3;
  sub_2180CE0((__int64)v48, a5, &v46);
  v11 = v49;
  if ( !v49 )
    return 1;
  if ( a8 > 0x32 )
    return 0;
  sub_1525B90(a6, &v45);
  v13 = v46;
  v14 = *(unsigned int *)(v46 + 40);
  if ( !(_DWORD)v14 )
    goto LABEL_19;
  v44 = v11;
  v15 = 0;
  v42 = v8;
  v16 = a4;
  v17 = 40 * v14;
  while ( 1 )
  {
    v18 = v15 + *(_QWORD *)(v13 + 32);
    if ( !*(_BYTE *)v18 && (*(_BYTE *)(v18 + 3) & 0x10) == 0 )
    {
      v19 = *(unsigned int *)(v16 + 24);
      v47 = *(_DWORD *)(v18 + 8);
      v20 = *(_QWORD *)(v16 + 8) + 4 * v19;
      v23 = sub_1DF91F0(v16, &v47, v48);
      v25 = v48[0];
      if ( !(_BYTE)v23 )
      {
        v21 = *(unsigned int *)(v16 + 24);
        v25 = *(_QWORD *)(v16 + 8) + 4 * v21;
      }
      if ( v20 == v25 )
      {
        v26 = sub_217E810(a1, v47, v21, v22, v23, v24);
        if ( !v26 )
        {
          sub_1525B90(a6, &v47);
          v29 = *((unsigned int *)a7 + 2);
          if ( !(_DWORD)v29 )
          {
            v30 = *(_DWORD **)a6;
            v31 = *(unsigned int *)(a6 + 8);
            v32 = *(_DWORD *)(a6 + 8);
            if ( v30 != &v30[v31] )
            {
              v41 = v15;
              v33 = *(_DWORD **)a6;
              v34 = &v30[v31];
              do
              {
                v35 = v33++;
                sub_1525B90((__int64)a7, v35);
              }
              while ( v34 != v33 );
              v15 = v41;
              v32 = *(_DWORD *)(a6 + 8);
            }
            goto LABEL_25;
          }
          v32 = *(_DWORD *)(a6 + 8);
          if ( !v32 )
            return 0;
          v36 = 0;
          v37 = *a7;
          while ( *(_DWORD *)(v37 + 4LL * v36) == *(_DWORD *)(*(_QWORD *)a6 + 4LL * v36) )
          {
            if ( ++v36 == (_DWORD)v29 || v36 == v32 )
              goto LABEL_32;
          }
          --v36;
LABEL_32:
          if ( !v36 )
            return 0;
          v38 = v36;
          if ( v36 < v29 )
          {
LABEL_41:
            *((_DWORD *)a7 + 2) = v36;
            v32 = *(_DWORD *)(a6 + 8);
          }
          else if ( v36 > v29 )
          {
            if ( v36 > (unsigned __int64)*((unsigned int *)a7 + 3) )
            {
              sub_16CD150((__int64)a7, a7 + 2, v36, 4, v27, v28);
              v37 = *a7;
              v29 = *((unsigned int *)a7 + 2);
              v38 = v36;
            }
            v39 = (_DWORD *)(v37 + 4 * v29);
            for ( i = (_DWORD *)(v37 + 4 * v38); i != v39; ++v39 )
            {
              if ( v39 )
                *v39 = 0;
            }
            goto LABEL_41;
          }
LABEL_25:
          *(_DWORD *)(a6 + 8) = v32 - 1;
          goto LABEL_7;
        }
        if ( !(unsigned __int8)sub_2180DE0(a1, v26, v47, v16, v42, a6, (__int64)a7, a8 + 1) )
          return 0;
      }
    }
LABEL_7:
    v15 += 40;
    if ( v17 == v15 )
      break;
    v13 = v46;
  }
  v11 = v44;
LABEL_19:
  --*(_DWORD *)(a6 + 8);
  return v11;
}
