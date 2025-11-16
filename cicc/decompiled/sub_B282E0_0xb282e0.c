// Function: sub_B282E0
// Address: 0xb282e0
//
__int64 __fastcall sub_B282E0(
        __int64 a1,
        __int64 *a2,
        unsigned int a3,
        unsigned __int8 (__fastcall *a4)(__int64, __int64),
        unsigned int a5,
        __int64 *a6)
{
  unsigned int i; // eax
  __int64 v8; // rdx
  __int64 v9; // rbx
  int v10; // r13d
  __int64 v11; // r12
  __int64 v12; // rax
  int v13; // eax
  __int64 v14; // rax
  __int64 *v15; // r13
  __int64 *v16; // r12
  unsigned __int64 v17; // r15
  __int64 v18; // r14
  __int64 v19; // rax
  unsigned __int64 v20; // r10
  unsigned __int64 v21; // rdx
  __int64 *v22; // rax
  unsigned __int64 v24; // r12
  __int64 *v25; // r13
  __int64 *v26; // r15
  unsigned __int64 v27; // rax
  __int64 *v28; // r13
  __int64 v29; // r12
  __int64 v30; // r15
  unsigned int v31; // r14d
  __int64 *v32; // [rsp+10h] [rbp-530h]
  unsigned __int64 v33; // [rsp+28h] [rbp-518h]
  __int64 *v35; // [rsp+38h] [rbp-508h]
  unsigned __int64 v36; // [rsp+58h] [rbp-4E8h]
  __int64 *v37; // [rsp+58h] [rbp-4E8h]
  __int64 *v40[4]; // [rsp+70h] [rbp-4D0h] BYREF
  __int64 *v41[4]; // [rsp+90h] [rbp-4B0h] BYREF
  __int64 *v42; // [rsp+B0h] [rbp-490h] BYREF
  unsigned int v43; // [rsp+B8h] [rbp-488h]
  char v44; // [rsp+C0h] [rbp-480h] BYREF
  _BYTE *v45; // [rsp+100h] [rbp-440h] BYREF
  unsigned int v46; // [rsp+108h] [rbp-438h]
  unsigned int v47; // [rsp+10Ch] [rbp-434h]
  _BYTE v48[1072]; // [rsp+110h] [rbp-430h] BYREF

  v42 = a2;
  v43 = a5;
  sub_B1C510(&v45, &v42, 1);
  *(_DWORD *)(sub_B20CA0(a1, (__int64)a2) + 4) = a5;
  for ( i = v46; v46; i = v46 )
  {
    while ( 1 )
    {
      v8 = (__int64)&v45[16 * i - 16];
      v9 = *(_QWORD *)v8;
      v10 = *(_DWORD *)(v8 + 8);
      v46 = i - 1;
      a2 = (__int64 *)v9;
      v11 = sub_B20CA0(a1, v9);
      v12 = *(unsigned int *)(v11 + 32);
      if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(v11 + 36) )
      {
        a2 = (__int64 *)(v11 + 40);
        sub_C8D5F0(v11 + 24, v11 + 40, v12 + 1, 4);
        v12 = *(unsigned int *)(v11 + 32);
      }
      *(_DWORD *)(*(_QWORD *)(v11 + 24) + 4 * v12) = v10;
      v13 = *(_DWORD *)v11;
      ++*(_DWORD *)(v11 + 32);
      if ( !v13 )
      {
        ++a3;
        *(_DWORD *)(v11 + 4) = v10;
        *(_DWORD *)(v11 + 12) = a3;
        *(_DWORD *)(v11 + 8) = a3;
        *(_DWORD *)v11 = a3;
        sub_B1A4E0(a1, v9);
        a2 = (__int64 *)v9;
        sub_B1CB80(&v42, v9, *(_QWORD *)(a1 + 4128));
        v14 = v43;
        if ( a6 && v43 > 1uLL )
        {
          v24 = v43;
          v25 = v42;
          v26 = &v42[v24];
          _BitScanReverse64(&v27, (__int64)(v24 * 8) >> 3);
          v32 = &v42[v24];
          sub_B27E00(v42, (char *)&v42[v24], 2LL * (int)(63 - (v27 ^ 0x3F)), (__int64)a6);
          if ( v24 <= 16 )
          {
            a2 = v32;
            sub_B23CD0(v25, v32, a6);
          }
          else
          {
            a2 = v25 + 16;
            v35 = v25 + 16;
            sub_B23CD0(v25, v25 + 16, a6);
            if ( v25 + 16 != v26 )
            {
              do
              {
                v28 = v35;
                v29 = *v35;
                while ( 1 )
                {
                  v30 = *(v28 - 1);
                  v37 = v28--;
                  sub_B1C5B0(v40, a6, v29);
                  a2 = a6;
                  v31 = *((_DWORD *)v40[2] + 2);
                  sub_B1C5B0(v41, a6, v30);
                  if ( v31 >= *((_DWORD *)v41[2] + 2) )
                    break;
                  v28[1] = *v28;
                }
                ++v35;
                *v37 = v29;
              }
              while ( v35 != v32 );
            }
          }
          v14 = v43;
        }
        v15 = v42;
        v16 = &v42[v14];
        if ( v16 != v42 )
        {
          v17 = v33;
          do
          {
            v18 = *v15;
            a2 = (__int64 *)*v15;
            if ( a4(v9, *v15) )
            {
              v19 = v46;
              v20 = v17 & 0xFFFFFFFF00000000LL | a3;
              v21 = v46 + 1LL;
              v17 = v20;
              if ( v21 > v47 )
              {
                a2 = (__int64 *)v48;
                v36 = v20;
                sub_C8D5F0(&v45, v48, v21, 16);
                v19 = v46;
                v20 = v36;
              }
              v22 = (__int64 *)&v45[16 * v19];
              *v22 = v18;
              v22[1] = v20;
              ++v46;
            }
            ++v15;
          }
          while ( v16 != v15 );
          v33 = v17;
          v15 = v42;
        }
        if ( v15 != (__int64 *)&v44 )
          break;
      }
      i = v46;
      if ( !v46 )
        goto LABEL_19;
    }
    _libc_free(v15, a2);
  }
LABEL_19:
  if ( v45 != v48 )
    _libc_free(v45, a2);
  return a3;
}
