// Function: sub_295CCA0
// Address: 0x295cca0
//
void __fastcall sub_295CCA0(unsigned __int64 *a1)
{
  unsigned __int64 v1; // r12
  unsigned __int64 v2; // r15
  __int64 v3; // r9
  _QWORD *v4; // rbx
  _QWORD *v5; // r14
  __int64 i; // rax
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rsi
  _QWORD *v10; // r14
  _QWORD *v11; // rbx
  __int64 v12; // rsi
  unsigned __int64 v13; // [rsp+18h] [rbp-98h]
  _QWORD v14[2]; // [rsp+28h] [rbp-88h] BYREF
  __int64 v15; // [rsp+38h] [rbp-78h]
  __int64 v16; // [rsp+40h] [rbp-70h]
  void *v17; // [rsp+50h] [rbp-60h]
  _QWORD v18[2]; // [rsp+58h] [rbp-58h] BYREF
  __int64 v19; // [rsp+68h] [rbp-48h]
  __int64 v20; // [rsp+70h] [rbp-40h]

  v1 = *a1 + 8LL * *((unsigned int *)a1 + 2);
  v13 = *a1;
  if ( *a1 != v1 )
  {
    do
    {
      v2 = *(_QWORD *)(v1 - 8);
      v1 -= 8LL;
      if ( v2 )
      {
        if ( *(_BYTE *)(v2 + 64) )
        {
          v9 = *(unsigned int *)(v2 + 56);
          *(_BYTE *)(v2 + 64) = 0;
          if ( (_DWORD)v9 )
          {
            v10 = *(_QWORD **)(v2 + 40);
            v11 = &v10[2 * v9];
            do
            {
              if ( *v10 != -4096 && *v10 != -8192 )
              {
                v12 = v10[1];
                if ( v12 )
                  sub_B91220((__int64)(v10 + 1), v12);
              }
              v10 += 2;
            }
            while ( v11 != v10 );
            v9 = *(unsigned int *)(v2 + 56);
          }
          sub_C7D6A0(*(_QWORD *)(v2 + 40), 16 * v9, 8);
        }
        if ( *(_DWORD *)(v2 + 24) )
        {
          v15 = -4096;
          v16 = 0;
          v19 = -8192;
          v20 = 0;
          v3 = *(unsigned int *)(v2 + 24);
          v14[0] = 2;
          v14[1] = 0;
          v3 <<= 6;
          v18[0] = 2;
          v18[1] = 0;
          v4 = *(_QWORD **)(v2 + 8);
          v5 = (_QWORD *)((char *)v4 + v3);
          v17 = &unk_49DD7B0;
          if ( v4 != (_QWORD *)((char *)v4 + v3) )
          {
            for ( i = -4096; ; i = v15 )
            {
              v7 = v4[3];
              if ( v7 != i )
              {
                i = v19;
                if ( v7 != v19 )
                {
                  v8 = v4[7];
                  if ( v8 != -4096 && v8 != 0 && v8 != -8192 )
                  {
                    sub_BD60C0(v4 + 5);
                    v7 = v4[3];
                  }
                  i = v7;
                }
              }
              *v4 = &unk_49DB368;
              if ( i != -4096 && i != 0 && i != -8192 )
                sub_BD60C0(v4 + 1);
              v4 += 8;
              if ( v5 == v4 )
                break;
            }
            v17 = &unk_49DB368;
            if ( v19 != 0 && v19 != -8192 && v19 != -4096 )
              sub_BD60C0(v18);
          }
          if ( v15 != -4096 && v15 != 0 && v15 != -8192 )
            sub_BD60C0(v14);
        }
        sub_C7D6A0(*(_QWORD *)(v2 + 8), (unsigned __int64)*(unsigned int *)(v2 + 24) << 6, 8);
        j_j___libc_free_0(v2);
      }
    }
    while ( v13 != v1 );
    v1 = *a1;
  }
  if ( (unsigned __int64 *)v1 != a1 + 2 )
    _libc_free(v1);
}
