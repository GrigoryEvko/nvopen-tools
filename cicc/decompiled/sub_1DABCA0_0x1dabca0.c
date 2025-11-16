// Function: sub_1DABCA0
// Address: 0x1dabca0
//
void __fastcall sub_1DABCA0(__int64 a1, unsigned int a2, unsigned int *a3, unsigned int *a4)
{
  signed int v4; // esi
  __int64 v6; // rdx
  unsigned int *v7; // r15
  unsigned int *i; // r14
  unsigned int v9; // r8d
  unsigned int v10; // r10d
  __int64 v11; // r12
  int v12; // r10d
  unsigned int v13; // esi
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rdx
  unsigned int v17; // r8d
  __int64 v18; // r11
  __int64 v19; // rdx
  __int64 v20; // r9
  _QWORD *v21; // rcx
  __int64 v22; // rsi
  unsigned int v23; // r8d
  __int64 v24; // rdx
  __int64 v25; // r9
  _QWORD *v26; // rcx
  unsigned int v27; // r10d
  unsigned int v28; // edx
  unsigned int v29; // r10d
  __int64 v30; // rdx
  unsigned int v31; // r11d
  __int64 v32; // r9
  _QWORD *v33; // rcx
  unsigned int v34; // esi
  __int64 j; // rdx
  __int64 v36; // rdi
  _QWORD *v37; // rcx
  unsigned int *v38; // r14
  __int64 v39; // r15
  unsigned int v40; // ecx
  unsigned int v41; // r13d
  unsigned int v42; // eax
  unsigned int *v43; // rbx
  int v44; // eax
  signed int v46; // [rsp+10h] [rbp-50h]
  unsigned int v49; // [rsp+20h] [rbp-40h]
  unsigned int *v50; // [rsp+20h] [rbp-40h]
  __int64 *v51; // [rsp+28h] [rbp-38h]
  unsigned int *v52; // [rsp+28h] [rbp-38h]

  v4 = a2 - 1;
  v46 = v4;
  if ( !v4 )
    return;
  v6 = v4;
  v51 = (__int64 *)(a1 + 8LL * v4);
  v7 = &a3[v6];
  for ( i = &a4[v6]; ; --i )
  {
    v9 = *v7;
    v10 = *i;
    --v46;
    if ( *v7 != *i )
    {
      v11 = v46;
      if ( v46 == -1 )
        goto LABEL_20;
      do
      {
        v12 = v10 - v9;
        v13 = a3[v11];
        v14 = *v51;
        v15 = *(_QWORD *)(a1 + 8 * v11);
        if ( v12 <= 0 )
        {
          v28 = 9 - v13;
          if ( 9 - v13 > v9 )
            v28 = v9;
          v29 = -v12;
          if ( v28 <= v29 )
            v29 = v28;
          v30 = 0;
          v31 = v29 + v13;
          if ( v29 )
          {
            do
            {
              v32 = v13++;
              v33 = (_QWORD *)(v15 + 16 * v32);
              *v33 = *(_QWORD *)(v14 + 4 * v30);
              v33[1] = *(_QWORD *)(v14 + 4 * v30 + 8);
              LODWORD(v33) = *(_DWORD *)(v14 + v30 + 144);
              v30 += 4;
              *(_DWORD *)(v15 + 4 * v32 + 144) = (_DWORD)v33;
            }
            while ( v31 != v13 );
          }
          v34 = v29;
          for ( j = 0; v9 != v34; j += 4 )
          {
            v36 = v34++;
            v37 = (_QWORD *)(v14 + 16 * v36);
            *(_QWORD *)(v14 + 4 * j) = *v37;
            *(_QWORD *)(v14 + 4 * j + 8) = v37[1];
            *(_DWORD *)(v14 + j + 144) = *(_DWORD *)(v14 + 4 * v36 + 144);
          }
          v12 = -v29;
        }
        else
        {
          if ( v12 > v13 )
            v12 = a3[v11];
          if ( 9 - v9 <= v12 )
            v12 = 9 - v9;
          v16 = v9 - 1;
          if ( v9 )
          {
            v49 = a3[v11];
            v17 = v16 + v12;
            v18 = -3 * v14;
            v19 = v14 + 4 * v16 + 144;
            do
            {
              v20 = v17--;
              v21 = (_QWORD *)(v14 + 16 * v20);
              *v21 = *(_QWORD *)(v18 + 4 * v19 - 576);
              v22 = *(_QWORD *)(v18 + 4 * v19 - 568);
              v19 -= 4;
              v21[1] = v22;
              *(_DWORD *)(v14 + 4 * v20 + 144) = *(_DWORD *)(v19 + 4);
            }
            while ( v14 + 140 != v19 );
            v13 = v49;
          }
          v23 = v13 - v12;
          if ( v13 != v13 - v12 )
          {
            v24 = 0;
            do
            {
              v25 = v23++;
              v26 = (_QWORD *)(v15 + 16 * v25);
              *(_QWORD *)(v14 + 4 * v24) = *v26;
              *(_QWORD *)(v14 + 4 * v24 + 8) = v26[1];
              *(_DWORD *)(v14 + v24 + 144) = *(_DWORD *)(v15 + 4 * v25 + 144);
              v24 += 4;
            }
            while ( v13 != v23 );
          }
        }
        a3[v11] -= v12;
        v27 = *v7 + v12;
        *v7 = v27;
        v9 = v27;
        v10 = *i;
        if ( v9 >= *i )
          break;
        --v11;
      }
      while ( (_DWORD)v11 != -1 );
    }
    if ( !v46 )
      break;
LABEL_20:
    --v51;
    --v7;
  }
  if ( a2 > 1 )
  {
    v38 = a3;
    v39 = 1;
    v50 = a3;
    v52 = a4;
    do
    {
      v40 = *v38;
      v41 = v39;
      v42 = *v52;
      if ( *v38 != *v52 && a2 != (_DWORD)v39 )
      {
        do
        {
          v43 = &v50[v41];
          v44 = sub_1DABB00(*(_QWORD *)(a1 + 8LL * v41), *v43, *(_QWORD *)(a1 + 8 * v39 - 8), v40, v40 - v42);
          *v43 += v44;
          v40 = *v38 - v44;
          *v38 = v40;
          v42 = *v52;
          if ( v40 >= *v52 )
            break;
          ++v41;
        }
        while ( a2 != v41 );
      }
      ++v52;
      ++v39;
      ++v38;
    }
    while ( v39 != a2 );
  }
}
