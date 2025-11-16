// Function: sub_1A53B20
// Address: 0x1a53b20
//
__int64 __fastcall sub_1A53B20(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rax
  __int64 v6; // rax
  _QWORD *v7; // rdx
  __int64 v8; // rcx
  _QWORD *v9; // rax
  _QWORD *v10; // rcx
  __int64 v11; // rbx
  __int64 v12; // r14
  __int64 v13; // rsi
  _QWORD *v14; // r15
  _QWORD *v15; // r13
  __int64 i; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v20; // rax
  _QWORD *v21; // r13
  _QWORD *v22; // r15
  __int64 v23; // rsi
  int v24; // [rsp+8h] [rbp-B8h]
  __int64 v25; // [rsp+10h] [rbp-B0h]
  unsigned __int64 v26; // [rsp+28h] [rbp-98h]
  _QWORD v27[2]; // [rsp+38h] [rbp-88h] BYREF
  __int64 v28; // [rsp+48h] [rbp-78h]
  __int64 v29; // [rsp+50h] [rbp-70h]
  void *v30; // [rsp+60h] [rbp-60h]
  _QWORD v31[2]; // [rsp+68h] [rbp-58h] BYREF
  __int64 v32; // [rsp+78h] [rbp-48h]
  __int64 v33; // [rsp+80h] [rbp-40h]

  v2 = a2;
  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
  v3 = ((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
      | (*(unsigned int *)(a1 + 12) + 2LL)
      | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4;
  v4 = ((v3
       | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | v3
     | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v5 = (v4 | (v4 >> 16) | HIDWORD(v4)) + 1;
  if ( v5 >= a2 )
    v2 = v5;
  v6 = 0xFFFFFFFFLL;
  if ( v2 <= 0xFFFFFFFF )
    v6 = v2;
  v24 = v6;
  v25 = malloc(8 * v6);
  if ( !v25 )
    sub_16BD1C0("Allocation failed", 1u);
  v7 = *(_QWORD **)a1;
  v8 = 8LL * *(unsigned int *)(a1 + 8);
  v26 = *(_QWORD *)a1 + v8;
  if ( *(_QWORD *)a1 != v26 )
  {
    v9 = (_QWORD *)v25;
    v10 = (_QWORD *)(v25 + v8);
    do
    {
      if ( v9 )
      {
        *v9 = *v7;
        *v7 = 0;
      }
      ++v9;
      ++v7;
    }
    while ( v9 != v10 );
    v26 = *(_QWORD *)a1;
    v11 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    if ( v11 != *(_QWORD *)a1 )
    {
      do
      {
        v12 = *(_QWORD *)(v11 - 8);
        v11 -= 8;
        if ( v12 )
        {
          if ( *(_BYTE *)(v12 + 64) )
          {
            v20 = *(unsigned int *)(v12 + 56);
            if ( (_DWORD)v20 )
            {
              v21 = *(_QWORD **)(v12 + 40);
              v22 = &v21[2 * v20];
              do
              {
                if ( *v21 != -8 && *v21 != -4 )
                {
                  v23 = v21[1];
                  if ( v23 )
                    sub_161E7C0((__int64)(v21 + 1), v23);
                }
                v21 += 2;
              }
              while ( v22 != v21 );
            }
            j___libc_free_0(*(_QWORD *)(v12 + 40));
          }
          if ( *(_DWORD *)(v12 + 24) )
          {
            v28 = -8;
            v29 = 0;
            v32 = -16;
            v33 = 0;
            v13 = *(unsigned int *)(v12 + 24);
            v27[0] = 2;
            v27[1] = 0;
            v13 <<= 6;
            v31[0] = 2;
            v31[1] = 0;
            v14 = *(_QWORD **)(v12 + 8);
            v15 = (_QWORD *)((char *)v14 + v13);
            v30 = &unk_49E6B50;
            if ( v14 != (_QWORD *)((char *)v14 + v13) )
            {
              for ( i = -8; ; i = v28 )
              {
                v17 = v14[3];
                if ( v17 != i )
                {
                  i = v32;
                  if ( v17 != v32 )
                  {
                    v18 = v14[7];
                    if ( v18 != -8 && v18 != 0 && v18 != -16 )
                    {
                      sub_1649B30(v14 + 5);
                      v17 = v14[3];
                    }
                    i = v17;
                  }
                }
                *v14 = &unk_49EE2B0;
                if ( i != -8 && i != 0 && i != -16 )
                  sub_1649B30(v14 + 1);
                v14 += 8;
                if ( v15 == v14 )
                  break;
              }
              v30 = &unk_49EE2B0;
              if ( v32 != -16 && v32 != 0 && v32 != -8 )
                sub_1649B30(v31);
            }
            if ( v28 != 0 && v28 != -8 && v28 != -16 )
              sub_1649B30(v27);
          }
          j___libc_free_0(*(_QWORD *)(v12 + 8));
          j_j___libc_free_0(v12, 80);
        }
      }
      while ( v26 != v11 );
      v26 = *(_QWORD *)a1;
    }
  }
  if ( v26 != a1 + 16 )
    _libc_free(v26);
  *(_QWORD *)a1 = v25;
  *(_DWORD *)(a1 + 12) = v24;
  return a1;
}
