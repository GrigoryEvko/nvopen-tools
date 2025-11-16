// Function: sub_2908890
// Address: 0x2908890
//
__int64 __fastcall sub_2908890(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r13
  __int64 v5; // r14
  unsigned int v6; // edi
  __int64 v7; // rax
  unsigned __int64 *v8; // r15
  __int64 v9; // rdx
  __int64 v10; // r12
  unsigned __int64 *v11; // r13
  __int64 v12; // rax
  bool v13; // zf
  __int64 v14; // r13
  __int64 i; // rax
  __int64 v16; // rsi
  int v17; // eax
  int v18; // eax
  __int64 v19; // r9
  int v20; // edx
  __int64 v21; // r10
  __int64 v22; // rdi
  unsigned int v23; // esi
  __int64 v24; // r15
  __int64 v25; // r8
  __int64 result; // rax
  __int64 v27; // rbx
  unsigned __int64 *v28; // rbx
  __int64 v29; // [rsp+8h] [rbp-C8h]
  _QWORD v30[2]; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v31; // [rsp+30h] [rbp-A0h]
  _QWORD v32[2]; // [rsp+40h] [rbp-90h] BYREF
  __int64 v33; // [rsp+50h] [rbp-80h]
  _QWORD v34[4]; // [rsp+60h] [rbp-70h] BYREF
  __int64 v35; // [rsp+80h] [rbp-50h] BYREF
  __int64 v36; // [rsp+88h] [rbp-48h]
  __int64 v37; // [rsp+90h] [rbp-40h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  v7 = sub_C7D670(32LL * v6, 8);
  *(_QWORD *)(a1 + 8) = v7;
  v8 = (unsigned __int64 *)v7;
  if ( v5 )
  {
    v9 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v29 = 32 * v4;
    v10 = v5 + 32 * v4;
    v9 *= 32;
    v35 = 0;
    v11 = (unsigned __int64 *)(v7 + v9);
    v36 = 0;
    v37 = -4096;
    if ( v7 != v7 + v9 )
    {
      do
      {
        if ( v8 )
        {
          *v8 = 0;
          v8[1] = 0;
          v12 = v37;
          v13 = v37 == -4096;
          v8[2] = v37;
          if ( v12 != 0 && !v13 && v12 != -8192 )
            sub_BD6050(v8, v35 & 0xFFFFFFFFFFFFFFF8LL);
        }
        v8 += 4;
      }
      while ( v11 != v8 );
    }
    sub_D68D70(&v35);
    v33 = -8192;
    v30[0] = 0;
    v30[1] = 0;
    v31 = -4096;
    v32[0] = 0;
    v32[1] = 0;
    if ( v10 != v5 )
    {
      v14 = v5;
      for ( i = -4096; ; i = v31 )
      {
        v16 = *(_QWORD *)(v14 + 16);
        if ( i != v16 )
        {
          i = v33;
          if ( v16 != v33 )
          {
            v17 = *(_DWORD *)(a1 + 24);
            if ( !v17 )
            {
              sub_FC7530(0, v16);
              MEMORY[0x18] = 0;
              BUG();
            }
            v34[2] = -4096;
            v18 = v17 - 1;
            v19 = *(_QWORD *)(a1 + 8);
            v20 = 1;
            v35 = 0;
            v21 = 0;
            v36 = 0;
            v37 = -8192;
            v34[0] = 0;
            v34[1] = 0;
            v22 = *(_QWORD *)(v14 + 16);
            v23 = v18 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
            v24 = v19 + 32LL * v23;
            v25 = *(_QWORD *)(v24 + 16);
            if ( v25 != v22 )
            {
              while ( v25 != -4096 )
              {
                if ( !v21 && v25 == -8192 )
                  v21 = v24;
                v23 = v18 & (v20 + v23);
                v24 = v19 + 32LL * v23;
                v25 = *(_QWORD *)(v24 + 16);
                if ( v22 == v25 )
                  goto LABEL_17;
                ++v20;
              }
              if ( v21 )
                v24 = v21;
            }
LABEL_17:
            sub_D68D70(&v35);
            sub_D68D70(v34);
            sub_FC7530((_QWORD *)v24, *(_QWORD *)(v14 + 16));
            *(_DWORD *)(v24 + 24) = *(_DWORD *)(v14 + 24);
            ++*(_DWORD *)(a1 + 16);
            i = *(_QWORD *)(v14 + 16);
          }
        }
        if ( i != 0 && i != -4096 && i != -8192 )
          sub_BD60C0((_QWORD *)v14);
        v14 += 32;
        if ( v10 == v14 )
          break;
      }
    }
    sub_D68D70(v32);
    sub_D68D70(v30);
    return sub_C7D6A0(v5, v29, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    v27 = *(unsigned int *)(a1 + 24);
    v35 = 0;
    v36 = 0;
    v28 = (unsigned __int64 *)(v7 + 32 * v27);
    v37 = -4096;
    result = -4096;
    if ( v8 != v28 )
    {
      do
      {
        if ( v8 )
        {
          *v8 = 0;
          v8[1] = 0;
          v8[2] = result;
          if ( result != 0 && result != -4096 && result != -8192 )
          {
            sub_BD6050(v8, v35 & 0xFFFFFFFFFFFFFFF8LL);
            result = v37;
          }
        }
        v8 += 4;
      }
      while ( v28 != v8 );
      if ( result != -4096 && result != 0 && result != -8192 )
        return sub_BD60C0(&v35);
    }
  }
  return result;
}
