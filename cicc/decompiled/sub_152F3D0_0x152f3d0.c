// Function: sub_152F3D0
// Address: 0x152f3d0
//
void __fastcall sub_152F3D0(_DWORD *a1, unsigned int a2, __int64 a3, unsigned int a4)
{
  unsigned int v6; // r13d
  __int64 v7; // rdx
  __int64 v8; // r15
  unsigned __int64 v9; // r9
  unsigned int v10; // eax
  unsigned int v11; // r13d
  unsigned int v12; // ecx
  int v13; // eax
  unsigned int v14; // r10d
  unsigned int v15; // r10d
  int v16; // eax
  __int64 v17; // rax
  __int64 v18; // rdx
  int v19; // edx
  unsigned int v20; // ecx
  int v21; // r13d
  __int64 v22; // rdi
  __int64 v23; // rdx
  int v24; // edx
  unsigned int v25; // eax
  unsigned int v26; // ecx
  __int64 v27; // [rsp+0h] [rbp-60h]
  unsigned __int64 v28; // [rsp+8h] [rbp-58h]
  unsigned int v29; // [rsp+14h] [rbp-4Ch]
  unsigned int v30; // [rsp+14h] [rbp-4Ch]
  __int64 v31; // [rsp+18h] [rbp-48h]
  unsigned int v32; // [rsp+28h] [rbp-38h] BYREF
  char v33; // [rsp+2Ch] [rbp-34h]

  v6 = *(_DWORD *)(a3 + 8);
  if ( a4 )
  {
    v7 = *(_QWORD *)a3;
    v33 = 1;
    v32 = a2;
    sub_152A250((__int64)a1, a4, v7, v6, 0, 0, (__int64)&v32);
  }
  else
  {
    v8 = 0;
    sub_1524D80(a1, 3u, a1[4]);
    sub_1524E40(a1, a2, 6);
    sub_1524E40(a1, v6, 6);
    v27 = 8LL * v6;
    if ( v6 )
    {
      do
      {
        v9 = *(_QWORD *)(*(_QWORD *)a3 + v8);
        v10 = v9;
        if ( v9 == (unsigned int)v9 )
        {
          sub_1524E40(a1, v9, 6);
        }
        else
        {
          v11 = a1[3];
          v12 = a1[2];
          if ( v9 > 0x1F )
          {
            do
            {
              v15 = v9 & 0x1F | 0x20;
              v16 = v15 << v12;
              v12 += 6;
              v11 |= v16;
              a1[3] = v11;
              if ( v12 > 0x1F )
              {
                v17 = *(_QWORD *)a1;
                v18 = *(unsigned int *)(*(_QWORD *)a1 + 8LL);
                if ( (unsigned __int64)*(unsigned int *)(*(_QWORD *)a1 + 12LL) - v18 <= 3 )
                {
                  v28 = v9;
                  v29 = v9 & 0x1F | 0x20;
                  v31 = *(_QWORD *)a1;
                  sub_16CD150(*(_QWORD *)a1, v17 + 16, v18 + 4, 1);
                  v17 = v31;
                  v9 = v28;
                  v15 = v29;
                  v18 = *(unsigned int *)(v31 + 8);
                }
                *(_DWORD *)(*(_QWORD *)v17 + v18) = v11;
                v11 = 0;
                *(_DWORD *)(v17 + 8) += 4;
                v13 = a1[2];
                v14 = v15 >> (32 - v13);
                if ( v13 )
                  v11 = v14;
                v12 = ((_BYTE)v13 + 6) & 0x1F;
                a1[3] = v11;
              }
              v9 >>= 5;
              a1[2] = v12;
            }
            while ( v9 > 0x1F );
            v10 = v9;
          }
          v19 = v10 << v12;
          v20 = v12 + 6;
          v21 = v19 | v11;
          a1[3] = v21;
          if ( v20 > 0x1F )
          {
            v22 = *(_QWORD *)a1;
            v23 = *(unsigned int *)(*(_QWORD *)a1 + 8LL);
            if ( (unsigned __int64)*(unsigned int *)(*(_QWORD *)a1 + 12LL) - v23 <= 3 )
            {
              v30 = v10;
              sub_16CD150(v22, v22 + 16, v23 + 4, 1);
              v10 = v30;
              v23 = *(unsigned int *)(v22 + 8);
            }
            *(_DWORD *)(*(_QWORD *)v22 + v23) = v21;
            *(_DWORD *)(v22 + 8) += 4;
            v24 = a1[2];
            v25 = v10 >> (32 - v24);
            v26 = 0;
            if ( v24 )
              v26 = v25;
            a1[3] = v26;
            a1[2] = ((_BYTE)v24 + 6) & 0x1F;
          }
          else
          {
            a1[2] = v20;
          }
        }
        v8 += 8;
      }
      while ( v27 != v8 );
    }
  }
}
