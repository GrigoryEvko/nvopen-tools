// Function: sub_3953E20
// Address: 0x3953e20
//
__int64 __fastcall sub_3953E20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rdx
  __int64 v7; // rsi
  unsigned int v8; // ecx
  __int64 *v9; // rax
  __int64 v10; // r9
  __int64 v11; // rbx
  __int64 result; // rax
  __int64 v13; // rdx
  __int64 v14; // r15
  __int64 v15; // r11
  __int64 v16; // rax
  char v17; // di
  unsigned int v18; // esi
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rdx
  unsigned int v24; // esi
  unsigned int v25; // r9d
  __int64 v26; // r8
  unsigned int v27; // ecx
  __int64 *v28; // rax
  __int64 v29; // rdi
  unsigned int v30; // ecx
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // r10
  int v34; // r10d
  __int64 v35; // r10
  __int64 *v36; // rax
  __int64 *v37; // r10
  int v38; // eax
  int v39; // eax
  int v40; // eax
  int v41; // r10d
  int v42; // eax
  __int64 v43; // r9
  unsigned int v44; // ecx
  __int64 v45; // r8
  int v46; // edi
  __int64 *v47; // rsi
  int v48; // [rsp+8h] [rbp-58h]
  int v49; // [rsp+8h] [rbp-58h]
  __int64 v50; // [rsp+8h] [rbp-58h]
  __int64 v51; // [rsp+10h] [rbp-50h]
  __int64 v52; // [rsp+18h] [rbp-48h]
  unsigned int v53; // [rsp+18h] [rbp-48h]
  int v54; // [rsp+18h] [rbp-48h]
  __int64 v55; // [rsp+18h] [rbp-48h]
  __int64 *v56; // [rsp+18h] [rbp-48h]
  __int64 v57; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v58[7]; // [rsp+28h] [rbp-38h] BYREF

  v6 = *(unsigned int *)(a1 + 136);
  v7 = *(_QWORD *)(a1 + 120);
  if ( (_DWORD)v6 )
  {
    v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (__int64 *)(v7 + 16LL * v8);
    v10 = *v9;
    if ( a2 == *v9 )
      goto LABEL_3;
    v40 = 1;
    while ( v10 != -8 )
    {
      v41 = v40 + 1;
      v8 = (v6 - 1) & (v40 + v8);
      v9 = (__int64 *)(v7 + 16LL * v8);
      v10 = *v9;
      if ( a2 == *v9 )
        goto LABEL_3;
      v40 = v41;
    }
  }
  v9 = (__int64 *)(v7 + 16 * v6);
LABEL_3:
  v52 = v9[1];
  v11 = sub_157F280(a3);
  result = a1 + 56;
  v14 = v13;
  v51 = a1 + 56;
  if ( v11 != v13 )
  {
    v15 = v52;
    do
    {
      v16 = 0x17FFFFFFE8LL;
      v17 = *(_BYTE *)(v11 + 23) & 0x40;
      v18 = *(_DWORD *)(v11 + 20) & 0xFFFFFFF;
      if ( v18 )
      {
        v19 = 24LL * *(unsigned int *)(v11 + 56) + 8;
        v20 = 0;
        do
        {
          v21 = v11 - 24LL * v18;
          if ( v17 )
            v21 = *(_QWORD *)(v11 - 8);
          if ( a2 == *(_QWORD *)(v21 + v19) )
          {
            v16 = 24 * v20;
            goto LABEL_12;
          }
          ++v20;
          v19 += 8;
        }
        while ( v18 != (_DWORD)v20 );
        v16 = 0x17FFFFFFE8LL;
      }
LABEL_12:
      if ( v17 )
        v22 = *(_QWORD *)(v11 - 8);
      else
        v22 = v11 - 24LL * v18;
      v23 = *(_QWORD *)(v22 + v16);
      v24 = *(_DWORD *)(a1 + 80);
      v57 = v23;
      if ( v24 )
      {
        v25 = v24 - 1;
        v26 = *(_QWORD *)(a1 + 64);
        v27 = (v24 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v28 = (__int64 *)(v26 + 16LL * v27);
        v29 = *v28;
        if ( v23 == *v28 )
        {
          v30 = *((_DWORD *)v28 + 2);
LABEL_17:
          v31 = 1LL << v30;
          v32 = 8LL * (v30 >> 6);
LABEL_18:
          *(_QWORD *)(*(_QWORD *)(v15 + 48) + v32) |= v31;
        }
        else
        {
          v53 = (v24 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
          v33 = *v28;
          v48 = 1;
          while ( v33 != -8 )
          {
            v34 = v48++;
            v35 = v25 & (v34 + v53);
            v53 = v35;
            v33 = *(_QWORD *)(v26 + 16 * v35);
            if ( v23 == v33 )
            {
              v49 = 1;
              v36 = (__int64 *)(v26 + 16LL * (v25 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4))));
              v37 = 0;
              while ( v29 != -8 )
              {
                if ( v37 || v29 != -16 )
                  v36 = v37;
                v27 = v25 & (v49 + v27);
                v56 = (__int64 *)(v26 + 16LL * v27);
                v29 = *v56;
                if ( v23 == *v56 )
                {
                  v30 = *((_DWORD *)v56 + 2);
                  goto LABEL_17;
                }
                ++v49;
                v37 = v36;
                v36 = (__int64 *)(v26 + 16LL * v27);
              }
              if ( !v37 )
                v37 = v36;
              v38 = *(_DWORD *)(a1 + 72);
              ++*(_QWORD *)(a1 + 56);
              v39 = v38 + 1;
              if ( 4 * v39 >= 3 * v24 )
              {
                v50 = v15;
                sub_1BFE340(v51, 2 * v24);
                v42 = *(_DWORD *)(a1 + 80);
                if ( !v42 )
                {
                  ++*(_DWORD *)(a1 + 72);
                  BUG();
                }
                v23 = v57;
                v43 = *(_QWORD *)(a1 + 64);
                v54 = v42 - 1;
                v15 = v50;
                v44 = (v42 - 1) & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
                v39 = *(_DWORD *)(a1 + 72) + 1;
                v37 = (__int64 *)(v43 + 16LL * v44);
                v45 = *v37;
                if ( *v37 != v57 )
                {
                  v46 = 1;
                  v47 = 0;
                  while ( v45 != -8 )
                  {
                    if ( !v47 && v45 == -16 )
                      v47 = v37;
                    v44 = v54 & (v46 + v44);
                    v37 = (__int64 *)(v43 + 16LL * v44);
                    v45 = *v37;
                    if ( v57 == *v37 )
                      goto LABEL_36;
                    ++v46;
                  }
                  if ( v47 )
                    v37 = v47;
                }
              }
              else if ( v24 - *(_DWORD *)(a1 + 76) - v39 <= v24 >> 3 )
              {
                v55 = v15;
                sub_1BFE340(v51, v24);
                sub_1BFD9C0(v51, &v57, v58);
                v37 = (__int64 *)v58[0];
                v23 = v57;
                v15 = v55;
                v39 = *(_DWORD *)(a1 + 72) + 1;
              }
LABEL_36:
              *(_DWORD *)(a1 + 72) = v39;
              if ( *v37 != -8 )
                --*(_DWORD *)(a1 + 76);
              *v37 = v23;
              v32 = 0;
              v31 = 1;
              *((_DWORD *)v37 + 2) = 0;
              goto LABEL_18;
            }
          }
        }
      }
      result = *(_QWORD *)(v11 + 32);
      if ( !result )
        BUG();
      v11 = 0;
      if ( *(_BYTE *)(result - 8) == 77 )
        v11 = result - 24;
    }
    while ( v14 != v11 );
  }
  return result;
}
