// Function: sub_1FB18E0
// Address: 0x1fb18e0
//
__int64 __fastcall sub_1FB18E0(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // r15
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rbx
  _QWORD *v10; // r8
  __int64 v11; // r10
  unsigned int v12; // edi
  _QWORD *v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r13
  unsigned int v16; // esi
  int v17; // r9d
  int v18; // ecx
  int v19; // ecx
  __int64 v20; // r10
  unsigned int v21; // esi
  int v22; // eax
  _QWORD *v23; // rdx
  __int64 v24; // rdi
  int v25; // r14d
  _QWORD *v26; // r11
  __int64 v27; // rsi
  __int64 result; // rax
  int v29; // eax
  __int64 v30; // rax
  int v31; // ecx
  int v32; // ecx
  _QWORD *v33; // r10
  unsigned int v34; // r14d
  __int64 v35; // rdi
  int v36; // r11d
  __int64 v37; // rsi
  _QWORD *v38; // [rsp+0h] [rbp-60h]
  _QWORD *v39; // [rsp+0h] [rbp-60h]
  int v40; // [rsp+8h] [rbp-58h]
  int v41; // [rsp+8h] [rbp-58h]
  _QWORD *v42; // [rsp+8h] [rbp-58h]
  int v43; // [rsp+8h] [rbp-58h]
  __int64 (__fastcall **v44)(); // [rsp+10h] [rbp-50h] BYREF
  __int64 v45; // [rsp+18h] [rbp-48h]
  __int64 v46; // [rsp+20h] [rbp-40h]
  __int64 v47; // [rsp+28h] [rbp-38h]

  v2 = a2;
  v4 = *(_QWORD *)a1;
  v47 = a1;
  v5 = *(_QWORD *)(v4 + 664);
  v46 = v4;
  v45 = v5;
  *(_QWORD *)(v4 + 664) = &v44;
  v6 = a2[4];
  v7 = a2[2];
  v8 = v2[3];
  v44 = off_49FFF30;
  sub_1D44C70(*(_QWORD *)a1, v7, v8, v6, v2[5]);
  sub_1F81BC0(a1, v2[4]);
  v9 = *(_QWORD *)(v2[4] + 48LL);
  if ( v9 )
  {
    v10 = v2;
    while ( 1 )
    {
      v15 = *(_QWORD *)(v9 + 16);
      if ( *(_WORD *)(v15 + 24) != 212 )
      {
        v16 = *(_DWORD *)(a1 + 584);
        v17 = *(_DWORD *)(a1 + 40);
        if ( !v16 )
        {
          ++*(_QWORD *)(a1 + 560);
          goto LABEL_8;
        }
        v11 = *(_QWORD *)(a1 + 568);
        v12 = (v16 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
        v13 = (_QWORD *)(v11 + 16LL * v12);
        v14 = *v13;
        if ( v15 != *v13 )
        {
          v41 = 1;
          v23 = 0;
          while ( v14 != -8 )
          {
            if ( v14 != -16 || v23 )
              v13 = v23;
            v12 = (v16 - 1) & (v41 + v12);
            v14 = *(_QWORD *)(v11 + 16LL * v12);
            if ( v15 == v14 )
              goto LABEL_4;
            ++v41;
            v23 = v13;
            v13 = (_QWORD *)(v11 + 16LL * v12);
          }
          if ( !v23 )
            v23 = v13;
          v29 = *(_DWORD *)(a1 + 576);
          ++*(_QWORD *)(a1 + 560);
          v22 = v29 + 1;
          if ( 4 * v22 >= 3 * v16 )
          {
LABEL_8:
            v38 = v10;
            v40 = v17;
            sub_1D45DD0(a1 + 560, 2 * v16);
            v18 = *(_DWORD *)(a1 + 584);
            if ( !v18 )
              goto LABEL_51;
            v19 = v18 - 1;
            v17 = v40;
            v20 = *(_QWORD *)(a1 + 568);
            v10 = v38;
            v21 = v19 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
            v22 = *(_DWORD *)(a1 + 576) + 1;
            v23 = (_QWORD *)(v20 + 16LL * v21);
            v24 = *v23;
            if ( v15 != *v23 )
            {
              v25 = 1;
              v26 = 0;
              while ( v24 != -8 )
              {
                if ( v24 == -16 && !v26 )
                  v26 = v23;
                v21 = v19 & (v25 + v21);
                v23 = (_QWORD *)(v20 + 16LL * v21);
                v24 = *v23;
                if ( v15 == *v23 )
                  goto LABEL_25;
                ++v25;
              }
              if ( v26 )
                v23 = v26;
            }
          }
          else if ( v16 - *(_DWORD *)(a1 + 580) - v22 <= v16 >> 3 )
          {
            v39 = v10;
            v43 = v17;
            sub_1D45DD0(a1 + 560, v16);
            v31 = *(_DWORD *)(a1 + 584);
            if ( !v31 )
            {
LABEL_51:
              ++*(_DWORD *)(a1 + 576);
              BUG();
            }
            v32 = v31 - 1;
            v33 = 0;
            v17 = v43;
            v10 = v39;
            v34 = v32 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
            v35 = *(_QWORD *)(a1 + 568);
            v36 = 1;
            v22 = *(_DWORD *)(a1 + 576) + 1;
            v23 = (_QWORD *)(v35 + 16LL * v34);
            v37 = *v23;
            if ( v15 != *v23 )
            {
              while ( v37 != -8 )
              {
                if ( !v33 && v37 == -16 )
                  v33 = v23;
                v34 = v32 & (v36 + v34);
                v23 = (_QWORD *)(v35 + 16LL * v34);
                v37 = *v23;
                if ( v15 == *v23 )
                  goto LABEL_25;
                ++v36;
              }
              if ( v33 )
                v23 = v33;
            }
          }
LABEL_25:
          *(_DWORD *)(a1 + 576) = v22;
          if ( *v23 != -8 )
            --*(_DWORD *)(a1 + 580);
          *v23 = v15;
          *((_DWORD *)v23 + 2) = v17;
          v30 = *(unsigned int *)(a1 + 40);
          if ( (unsigned int)v30 >= *(_DWORD *)(a1 + 44) )
          {
            v42 = v10;
            sub_16CD150(a1 + 32, (const void *)(a1 + 48), 0, 8, (int)v10, v17);
            v30 = *(unsigned int *)(a1 + 40);
            v10 = v42;
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v30) = v15;
          ++*(_DWORD *)(a1 + 40);
        }
      }
LABEL_4:
      v9 = *(_QWORD *)(v9 + 32);
      if ( !v9 )
      {
        v2 = v10;
        break;
      }
    }
  }
  v27 = v2[2];
  if ( !*(_QWORD *)(v27 + 48) )
    sub_1F81E80((__int64 *)a1, v27);
  result = v46;
  *(_QWORD *)(v46 + 664) = v45;
  return result;
}
