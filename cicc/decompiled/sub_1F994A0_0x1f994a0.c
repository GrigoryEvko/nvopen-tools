// Function: sub_1F994A0
// Address: 0x1f994a0
//
__int64 __fastcall sub_1F994A0(__int64 a1, __int64 a2, __int64 *a3, int a4, char a5)
{
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // rbx
  int v13; // r11d
  _QWORD *v14; // r9
  _QWORD *v15; // rdx
  unsigned int v16; // edi
  _QWORD *v17; // rax
  __int64 v18; // rcx
  __int64 v19; // r13
  unsigned int v20; // esi
  int v21; // r8d
  int v22; // eax
  int v23; // esi
  __int64 v24; // rdi
  unsigned int v25; // eax
  int v26; // ecx
  __int64 v27; // rax
  int v29; // eax
  int v30; // eax
  int v31; // eax
  int v32; // r10d
  unsigned int v33; // r14d
  __int64 v34; // rdi
  __int64 v35; // rsi
  int v36; // r11d
  _QWORD *v37; // r10
  int v38; // [rsp+0h] [rbp-70h]
  int v39; // [rsp+0h] [rbp-70h]
  __int64 v40; // [rsp+8h] [rbp-68h]
  __int64 v42; // [rsp+18h] [rbp-58h]
  __int64 (__fastcall **v43)(); // [rsp+20h] [rbp-50h] BYREF
  __int64 v44; // [rsp+28h] [rbp-48h]
  __int64 v45; // [rsp+30h] [rbp-40h]
  __int64 v46; // [rsp+38h] [rbp-38h]

  v9 = *(_QWORD *)a1;
  v46 = a1;
  v10 = *(_QWORD *)(v9 + 664);
  v45 = v9;
  v44 = v10;
  *(_QWORD *)(v9 + 664) = &v43;
  v11 = *(_QWORD *)a1;
  v43 = off_49FFF30;
  sub_1D44A40(v11, a2, a3);
  if ( a5 && a4 )
  {
    v42 = (__int64)&a3[2 * (unsigned int)(a4 - 1) + 2];
    v40 = a1 + 560;
    while ( 1 )
    {
      if ( *a3 )
      {
        sub_1F81BC0(a1, *a3);
        v12 = *(_QWORD *)(*a3 + 48);
        if ( v12 )
          break;
      }
LABEL_19:
      a3 += 2;
      if ( (__int64 *)v42 == a3 )
        goto LABEL_20;
    }
    while ( 1 )
    {
      v19 = *(_QWORD *)(v12 + 16);
      if ( *(_WORD *)(v19 + 24) == 212 )
        goto LABEL_8;
      v20 = *(_DWORD *)(a1 + 584);
      v21 = *(_DWORD *)(a1 + 40);
      if ( !v20 )
      {
        ++*(_QWORD *)(a1 + 560);
        goto LABEL_12;
      }
      v13 = 1;
      v14 = *(_QWORD **)(a1 + 568);
      v15 = 0;
      v16 = (v20 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      v17 = &v14[2 * v16];
      v18 = *v17;
      if ( v19 == *v17 )
      {
LABEL_8:
        v12 = *(_QWORD *)(v12 + 32);
        if ( !v12 )
          goto LABEL_19;
      }
      else
      {
        while ( v18 != -8 )
        {
          if ( v18 != -16 || v15 )
            v17 = v15;
          v16 = (v20 - 1) & (v13 + v16);
          v18 = v14[2 * v16];
          if ( v19 == v18 )
            goto LABEL_8;
          ++v13;
          v15 = v17;
          v17 = &v14[2 * v16];
        }
        if ( !v15 )
          v15 = v17;
        v29 = *(_DWORD *)(a1 + 576);
        ++*(_QWORD *)(a1 + 560);
        v26 = v29 + 1;
        if ( 4 * (v29 + 1) < 3 * v20 )
        {
          if ( v20 - *(_DWORD *)(a1 + 580) - v26 <= v20 >> 3 )
          {
            v39 = v21;
            sub_1D45DD0(v40, v20);
            v30 = *(_DWORD *)(a1 + 584);
            if ( !v30 )
            {
LABEL_54:
              ++*(_DWORD *)(a1 + 576);
              BUG();
            }
            v31 = v30 - 1;
            v14 = 0;
            v21 = v39;
            v32 = 1;
            v33 = v31 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
            v34 = *(_QWORD *)(a1 + 568);
            v26 = *(_DWORD *)(a1 + 576) + 1;
            v15 = (_QWORD *)(v34 + 16LL * v33);
            v35 = *v15;
            if ( v19 != *v15 )
            {
              while ( v35 != -8 )
              {
                if ( !v14 && v35 == -16 )
                  v14 = v15;
                v33 = v31 & (v32 + v33);
                v15 = (_QWORD *)(v34 + 16LL * v33);
                v35 = *v15;
                if ( v19 == *v15 )
                  goto LABEL_14;
                ++v32;
              }
              if ( v14 )
                v15 = v14;
            }
          }
          goto LABEL_14;
        }
LABEL_12:
        v38 = v21;
        sub_1D45DD0(v40, 2 * v20);
        v22 = *(_DWORD *)(a1 + 584);
        if ( !v22 )
          goto LABEL_54;
        v23 = v22 - 1;
        v24 = *(_QWORD *)(a1 + 568);
        v21 = v38;
        v25 = (v22 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
        v26 = *(_DWORD *)(a1 + 576) + 1;
        v15 = (_QWORD *)(v24 + 16LL * v25);
        v14 = (_QWORD *)*v15;
        if ( v19 != *v15 )
        {
          v36 = 1;
          v37 = 0;
          while ( v14 != (_QWORD *)-8LL )
          {
            if ( v14 == (_QWORD *)-16LL && !v37 )
              v37 = v15;
            v25 = v23 & (v36 + v25);
            v15 = (_QWORD *)(v24 + 16LL * v25);
            v14 = (_QWORD *)*v15;
            if ( v19 == *v15 )
              goto LABEL_14;
            ++v36;
          }
          if ( v37 )
            v15 = v37;
        }
LABEL_14:
        *(_DWORD *)(a1 + 576) = v26;
        if ( *v15 != -8 )
          --*(_DWORD *)(a1 + 580);
        *v15 = v19;
        *((_DWORD *)v15 + 2) = v21;
        v27 = *(unsigned int *)(a1 + 40);
        if ( (unsigned int)v27 >= *(_DWORD *)(a1 + 44) )
        {
          sub_16CD150(a1 + 32, (const void *)(a1 + 48), 0, 8, v21, (int)v14);
          v27 = *(unsigned int *)(a1 + 40);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v27) = v19;
        ++*(_DWORD *)(a1 + 40);
        v12 = *(_QWORD *)(v12 + 32);
        if ( !v12 )
          goto LABEL_19;
      }
    }
  }
LABEL_20:
  if ( !*(_QWORD *)(a2 + 48) )
    sub_1F81E80((__int64 *)a1, a2);
  *(_QWORD *)(v45 + 664) = v44;
  return a2;
}
