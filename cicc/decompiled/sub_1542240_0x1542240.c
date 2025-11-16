// Function: sub_1542240
// Address: 0x1542240
//
void __fastcall sub_1542240(__int64 a1, unsigned int a2, unsigned int a3)
{
  unsigned int v4; // ebx
  __int64 v6; // r10
  __int64 v7; // r13
  __int64 v8; // r15
  __int64 v9; // r8
  unsigned int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // rdi
  unsigned int v13; // esi
  __int64 *v14; // r13
  int v15; // ecx
  int v16; // ecx
  __int64 v17; // r9
  int v18; // edx
  unsigned int v19; // esi
  __int64 v20; // r8
  __int64 v21; // rdx
  int v22; // r11d
  __int64 *v23; // r10
  int v24; // ecx
  int v25; // ecx
  int v26; // ecx
  __int64 v27; // r9
  __int64 *v28; // r10
  int v29; // r11d
  unsigned int v30; // esi
  __int64 v31; // r8
  int v32; // r11d
  __int64 **v33; // [rsp-70h] [rbp-70h]
  __int64 v34; // [rsp-68h] [rbp-68h]
  __int64 v35; // [rsp-60h] [rbp-60h]
  __int64 v36; // [rsp-58h] [rbp-58h] BYREF
  __int64 v37; // [rsp-50h] [rbp-50h]
  __int64 **v38; // [rsp-48h] [rbp-48h]

  if ( a2 != a3 )
  {
    v4 = a2;
    if ( a2 + 1 != a3 && !*(_BYTE *)(a1 + 320) )
    {
      v6 = *(_QWORD *)(a1 + 112);
      v7 = a2;
      v8 = 16LL * a2;
      v35 = 16LL * a3;
      v33 = (__int64 **)(v35 + v6);
      v34 = v6 + v8;
      sub_1540320(&v36, (__m128i *)(v6 + v8), (v35 - v8) >> 4);
      if ( v38 )
        sub_15411A0(v34, v33, v38, v37, a1);
      else
        sub_153D420(v34, v33, a1);
      j_j___libc_free_0(v38, 16 * v37);
      sub_1540570(
        (__m128i *)(v8 + *(_QWORD *)(a1 + 112)),
        (__int64 *)(*(_QWORD *)(a1 + 112) + v35),
        (unsigned __int8 (__fastcall *)(char *))sub_153C8E0);
      while ( 1 )
      {
        v13 = *(_DWORD *)(a1 + 104);
        v14 = (__int64 *)(*(_QWORD *)(a1 + 112) + 16 * v7);
        if ( v13 )
        {
          v9 = *(_QWORD *)(a1 + 88);
          v10 = (v13 - 1) & (((unsigned int)*v14 >> 9) ^ ((unsigned int)*v14 >> 4));
          v11 = (__int64 *)(v9 + 16LL * v10);
          v12 = *v11;
          if ( *v14 == *v11 )
            goto LABEL_9;
          v22 = 1;
          v23 = 0;
          while ( v12 != -8 )
          {
            if ( v12 == -16 && !v23 )
              v23 = v11;
            v10 = (v13 - 1) & (v22 + v10);
            v11 = (__int64 *)(v9 + 16LL * v10);
            v12 = *v11;
            if ( *v14 == *v11 )
              goto LABEL_9;
            ++v22;
          }
          v24 = *(_DWORD *)(a1 + 96);
          if ( v23 )
            v11 = v23;
          ++*(_QWORD *)(a1 + 80);
          v18 = v24 + 1;
          if ( 4 * (v24 + 1) < 3 * v13 )
          {
            if ( v13 - *(_DWORD *)(a1 + 100) - v18 > v13 >> 3 )
              goto LABEL_15;
            sub_1542080(a1 + 80, v13);
            v25 = *(_DWORD *)(a1 + 104);
            if ( !v25 )
            {
LABEL_49:
              ++*(_DWORD *)(a1 + 96);
              BUG();
            }
            v26 = v25 - 1;
            v27 = *(_QWORD *)(a1 + 88);
            v28 = 0;
            v29 = 1;
            v18 = *(_DWORD *)(a1 + 96) + 1;
            v30 = v26 & (((unsigned int)*v14 >> 9) ^ ((unsigned int)*v14 >> 4));
            v11 = (__int64 *)(v27 + 16LL * v30);
            v31 = *v11;
            if ( *v14 == *v11 )
              goto LABEL_15;
            while ( v31 != -8 )
            {
              if ( v31 == -16 && !v28 )
                v28 = v11;
              v30 = v26 & (v29 + v30);
              v11 = (__int64 *)(v27 + 16LL * v30);
              v31 = *v11;
              if ( *v14 == *v11 )
                goto LABEL_15;
              ++v29;
            }
            goto LABEL_28;
          }
        }
        else
        {
          ++*(_QWORD *)(a1 + 80);
        }
        sub_1542080(a1 + 80, 2 * v13);
        v15 = *(_DWORD *)(a1 + 104);
        if ( !v15 )
          goto LABEL_49;
        v16 = v15 - 1;
        v17 = *(_QWORD *)(a1 + 88);
        v18 = *(_DWORD *)(a1 + 96) + 1;
        v19 = v16 & (((unsigned int)*v14 >> 9) ^ ((unsigned int)*v14 >> 4));
        v11 = (__int64 *)(v17 + 16LL * v19);
        v20 = *v11;
        if ( *v11 == *v14 )
          goto LABEL_15;
        v32 = 1;
        v28 = 0;
        while ( v20 != -8 )
        {
          if ( !v28 && v20 == -16 )
            v28 = v11;
          v19 = v16 & (v32 + v19);
          v11 = (__int64 *)(v17 + 16LL * v19);
          v20 = *v11;
          if ( *v14 == *v11 )
            goto LABEL_15;
          ++v32;
        }
LABEL_28:
        if ( v28 )
          v11 = v28;
LABEL_15:
        *(_DWORD *)(a1 + 96) = v18;
        if ( *v11 != -8 )
          --*(_DWORD *)(a1 + 100);
        v21 = *v14;
        *((_DWORD *)v11 + 2) = 0;
        *v11 = v21;
LABEL_9:
        *((_DWORD *)v11 + 2) = ++v4;
        if ( a3 == v4 )
          return;
        v7 = v4;
      }
    }
  }
}
