// Function: sub_A1BFB0
// Address: 0xa1bfb0
//
void __fastcall sub_A1BFB0(__int64 a1, unsigned int a2, __int64 a3, unsigned int a4)
{
  unsigned int v6; // r13d
  __int64 v7; // r15
  unsigned __int64 v8; // r9
  unsigned int v9; // eax
  unsigned int v10; // r13d
  unsigned int v11; // ecx
  int v12; // eax
  unsigned int v13; // r10d
  unsigned int v14; // r10d
  int v15; // eax
  _QWORD *v16; // rax
  __int64 v17; // rdx
  int v18; // edx
  unsigned int v19; // ecx
  int v20; // r13d
  _QWORD *v21; // rdi
  __int64 v22; // rdx
  int v23; // edx
  unsigned int v24; // eax
  unsigned int v25; // ecx
  __int64 v26; // [rsp+0h] [rbp-50h]
  unsigned __int64 v27; // [rsp+8h] [rbp-48h]
  unsigned int v28; // [rsp+14h] [rbp-3Ch]
  unsigned int v29; // [rsp+14h] [rbp-3Ch]
  _QWORD *v30; // [rsp+18h] [rbp-38h]

  v6 = *(_DWORD *)(a3 + 8);
  if ( a4 )
  {
    sub_A1B020(a1, a4, *(_QWORD *)a3, v6, 0, 0, a2, 1);
  }
  else
  {
    v7 = 0;
    sub_A17B10(a1, 3u, *(_DWORD *)(a1 + 56));
    sub_A17CC0(a1, a2, 6);
    sub_A17CC0(a1, v6, 6);
    v26 = 8LL * v6;
    if ( v6 )
    {
      do
      {
        v8 = *(_QWORD *)(*(_QWORD *)a3 + v7);
        v9 = v8;
        if ( v8 == (unsigned int)v8 )
        {
          sub_A17CC0(a1, v8, 6);
        }
        else
        {
          v10 = *(_DWORD *)(a1 + 52);
          v11 = *(_DWORD *)(a1 + 48);
          if ( v8 > 0x1F )
          {
            do
            {
              v14 = v8 & 0x1F | 0x20;
              v15 = v14 << v11;
              v11 += 6;
              v10 |= v15;
              *(_DWORD *)(a1 + 52) = v10;
              if ( v11 > 0x1F )
              {
                v16 = *(_QWORD **)(a1 + 24);
                v17 = v16[1];
                if ( (unsigned __int64)(v17 + 4) > v16[2] )
                {
                  v27 = v8;
                  v28 = v8 & 0x1F | 0x20;
                  v30 = *(_QWORD **)(a1 + 24);
                  sub_C8D290(v30, v16 + 3, v17 + 4, 1);
                  v16 = v30;
                  v8 = v27;
                  v14 = v28;
                  v17 = v30[1];
                }
                *(_DWORD *)(*v16 + v17) = v10;
                v10 = 0;
                v16[1] += 4LL;
                v12 = *(_DWORD *)(a1 + 48);
                v13 = v14 >> (32 - v12);
                if ( v12 )
                  v10 = v13;
                v11 = ((_BYTE)v12 + 6) & 0x1F;
                *(_DWORD *)(a1 + 52) = v10;
              }
              v8 >>= 5;
              *(_DWORD *)(a1 + 48) = v11;
            }
            while ( v8 > 0x1F );
            v9 = v8;
          }
          v18 = v9 << v11;
          v19 = v11 + 6;
          v20 = v18 | v10;
          *(_DWORD *)(a1 + 52) = v20;
          if ( v19 > 0x1F )
          {
            v21 = *(_QWORD **)(a1 + 24);
            v22 = v21[1];
            if ( (unsigned __int64)(v22 + 4) > v21[2] )
            {
              v29 = v9;
              sub_C8D290(v21, v21 + 3, v22 + 4, 1);
              v9 = v29;
              v22 = v21[1];
            }
            *(_DWORD *)(*v21 + v22) = v20;
            v21[1] += 4LL;
            v23 = *(_DWORD *)(a1 + 48);
            v24 = v9 >> (32 - v23);
            v25 = 0;
            if ( v23 )
              v25 = v24;
            *(_DWORD *)(a1 + 52) = v25;
            *(_DWORD *)(a1 + 48) = ((_BYTE)v23 + 6) & 0x1F;
          }
          else
          {
            *(_DWORD *)(a1 + 48) = v19;
          }
        }
        v7 += 8;
      }
      while ( v26 != v7 );
    }
  }
}
