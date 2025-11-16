// Function: sub_1BE4A00
// Address: 0x1be4a00
//
__int64 __fastcall sub_1BE4A00(__int64 a1, __int64 a2, __int64 a3)
{
  const void *v3; // r15
  __int64 v7; // r8
  __int64 v8; // rax
  unsigned __int64 v9; // r9
  __int64 v10; // r10
  unsigned __int64 v11; // r14
  unsigned __int64 v12; // rdx
  unsigned int v13; // edx
  __int64 v14; // rax
  __int64 v15; // rdx
  int v16; // r9d
  __int64 v17; // rdi
  unsigned int v18; // ecx
  __int64 v19; // rsi
  __int64 v20; // r8
  _QWORD *v21; // r13
  _QWORD *v22; // r14
  unsigned __int64 v23; // rbx
  __int64 v24; // rax
  _QWORD *v25; // rdx
  _QWORD *v26; // rsi
  __int64 v27; // rcx
  __int64 v28; // rax
  _QWORD *v29; // rdi
  _QWORD *v30; // rax
  _QWORD *v31; // rax
  _QWORD *v32; // rcx
  int v34; // esi
  int v35; // r10d
  __int64 v36; // [rsp+8h] [rbp-48h]
  __int64 v37; // [rsp+10h] [rbp-40h]
  char *v38; // [rsp+18h] [rbp-38h]
  unsigned __int64 v39; // [rsp+18h] [rbp-38h]

  v3 = (const void *)(a1 + 16);
  v12 = *(unsigned int *)(a2 + 88);
  v7 = *(_QWORD *)(a2 + 80);
  *(_QWORD *)(a1 + 8) = 0x800000000LL;
  *(_QWORD *)a1 = a1 + 16;
  v8 = a1 + 16;
  v9 = v12;
  v10 = 8 * v12;
  v11 = v12;
  LODWORD(v12) = 0;
  if ( v9 > 8 )
  {
    v36 = v10;
    v37 = v7;
    v39 = v9;
    sub_16CD150(a1, (const void *)(a1 + 16), v9, 8, v7, v9);
    v12 = *(unsigned int *)(a1 + 8);
    v10 = v36;
    v7 = v37;
    v9 = v39;
    v8 = *(_QWORD *)a1 + 8 * v12;
  }
  if ( v10 )
  {
    do
    {
      v8 += 8;
      *(_QWORD *)(v8 - 8) = *(_QWORD *)(v7 + v10 - 8 * v9 + 8 * v11-- - 8);
    }
    while ( v11 );
    LODWORD(v12) = *(_DWORD *)(a1 + 8);
  }
  v13 = v9 + v12;
  *(_DWORD *)(a1 + 8) = v13;
  v14 = v13;
  if ( a3 )
  {
    v15 = *(unsigned int *)(a3 + 104);
    if ( (_DWORD)v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(a3 + 88);
      v18 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v19 = v17 + 56LL * v18;
      v20 = *(_QWORD *)v19;
      if ( a2 == *(_QWORD *)v19 )
      {
LABEL_9:
        if ( v19 != v17 + 56 * v15 )
        {
          v21 = *(_QWORD **)(v19 + 8);
          v22 = &v21[*(unsigned int *)(v19 + 16)];
          if ( v21 != v22 )
          {
            v23 = *v21 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*v21 & 4) != 0 )
              goto LABEL_28;
            while ( 1 )
            {
              v24 = 8 * v14;
              v25 = *(_QWORD **)a1;
              v26 = (_QWORD *)(*(_QWORD *)a1 + v24);
              v27 = v24 >> 3;
              v28 = v24 >> 5;
              if ( v28 )
              {
                v29 = *(_QWORD **)a1;
                v30 = &v25[4 * v28];
                while ( v23 != *v29 )
                {
                  if ( v23 == v29[1] )
                  {
                    ++v29;
                    goto LABEL_19;
                  }
                  if ( v23 == v29[2] )
                  {
                    v29 += 2;
                    goto LABEL_19;
                  }
                  if ( v23 == v29[3] )
                  {
                    v29 += 3;
                    goto LABEL_19;
                  }
                  v29 += 4;
                  if ( v29 == v30 )
                  {
                    v27 = v26 - v29;
                    goto LABEL_36;
                  }
                }
                goto LABEL_19;
              }
              v29 = *(_QWORD **)a1;
LABEL_36:
              if ( v27 == 2 )
                goto LABEL_42;
              if ( v27 != 3 )
              {
                if ( v27 != 1 )
                {
LABEL_39:
                  v32 = v26;
                  goto LABEL_26;
                }
LABEL_44:
                v32 = v26;
                if ( v23 != *v29 )
                  goto LABEL_26;
                goto LABEL_19;
              }
              if ( v23 != *v29 )
                break;
LABEL_19:
              if ( v26 == v29 )
                goto LABEL_39;
              v31 = v29 + 1;
              if ( v26 == v29 + 1 )
              {
                v32 = v29;
              }
              else
              {
                do
                {
                  if ( v23 != *v31 )
                    *v29++ = *v31;
                  ++v31;
                }
                while ( v26 != v31 );
                v25 = *(_QWORD **)a1;
                v20 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - (_QWORD)v26;
                v32 = (_QWORD *)((char *)v29 + v20);
                if ( v26 != (_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8)) )
                {
                  v38 = (char *)v29 + v20;
                  memmove(v29, v26, *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - (_QWORD)v26);
                  v25 = *(_QWORD **)a1;
                  v32 = v38;
                }
              }
LABEL_26:
              ++v21;
              *(_DWORD *)(a1 + 8) = v32 - v25;
              if ( v22 == v21 )
                return a1;
              while ( 1 )
              {
                v14 = *(unsigned int *)(a1 + 8);
                v23 = *v21 & 0xFFFFFFFFFFFFFFF8LL;
                if ( (*v21 & 4) == 0 )
                  break;
LABEL_28:
                if ( (unsigned int)v14 >= *(_DWORD *)(a1 + 12) )
                {
                  sub_16CD150(a1, v3, 0, 8, v20, v16);
                  v14 = *(unsigned int *)(a1 + 8);
                }
                ++v21;
                *(_QWORD *)(*(_QWORD *)a1 + 8 * v14) = v23;
                ++*(_DWORD *)(a1 + 8);
                if ( v22 == v21 )
                  return a1;
              }
            }
            ++v29;
LABEL_42:
            if ( v23 != *v29 )
            {
              ++v29;
              goto LABEL_44;
            }
            goto LABEL_19;
          }
        }
      }
      else
      {
        v34 = 1;
        while ( v20 != -8 )
        {
          v35 = v34 + 1;
          v18 = v16 & (v34 + v18);
          v19 = v17 + 56LL * v18;
          v20 = *(_QWORD *)v19;
          if ( a2 == *(_QWORD *)v19 )
            goto LABEL_9;
          v34 = v35;
        }
      }
    }
  }
  return a1;
}
