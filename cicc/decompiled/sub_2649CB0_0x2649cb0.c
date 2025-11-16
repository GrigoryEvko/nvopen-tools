// Function: sub_2649CB0
// Address: 0x2649cb0
//
void __fastcall sub_2649CB0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned __int64 v3; // r14
  __int64 v4; // rsi
  _QWORD *v5; // r15
  _QWORD *v6; // r12
  __int64 i; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rsi
  _QWORD *v11; // r13
  _QWORD *v12; // r15
  __int64 v13; // rsi
  _QWORD v14[2]; // [rsp+28h] [rbp-88h] BYREF
  __int64 v15; // [rsp+38h] [rbp-78h]
  __int64 v16; // [rsp+40h] [rbp-70h]
  void *v17; // [rsp+50h] [rbp-60h]
  _QWORD v18[2]; // [rsp+58h] [rbp-58h] BYREF
  __int64 v19; // [rsp+68h] [rbp-48h]
  __int64 v20; // [rsp+70h] [rbp-40h]

  if ( a2 != a1 )
  {
    v2 = a2;
    do
    {
      v3 = *(_QWORD *)(v2 - 8);
      v2 -= 8;
      if ( v3 )
      {
        if ( *(_BYTE *)(v3 + 64) )
        {
          v10 = *(unsigned int *)(v3 + 56);
          *(_BYTE *)(v3 + 64) = 0;
          if ( (_DWORD)v10 )
          {
            v11 = *(_QWORD **)(v3 + 40);
            v12 = &v11[2 * v10];
            do
            {
              if ( *v11 != -4096 && *v11 != -8192 )
              {
                v13 = v11[1];
                if ( v13 )
                  sub_B91220((__int64)(v11 + 1), v13);
              }
              v11 += 2;
            }
            while ( v12 != v11 );
            v10 = *(unsigned int *)(v3 + 56);
          }
          sub_C7D6A0(*(_QWORD *)(v3 + 40), 16 * v10, 8);
        }
        if ( *(_DWORD *)(v3 + 24) )
        {
          v15 = -4096;
          v16 = 0;
          v19 = -8192;
          v20 = 0;
          v4 = *(unsigned int *)(v3 + 24);
          v14[0] = 2;
          v14[1] = 0;
          v4 <<= 6;
          v18[0] = 2;
          v18[1] = 0;
          v5 = *(_QWORD **)(v3 + 8);
          v6 = (_QWORD *)((char *)v5 + v4);
          v17 = &unk_49DD7B0;
          if ( v5 != (_QWORD *)((char *)v5 + v4) )
          {
            for ( i = -4096; ; i = v15 )
            {
              v8 = v5[3];
              if ( v8 != i )
              {
                i = v19;
                if ( v8 != v19 )
                {
                  v9 = v5[7];
                  if ( v9 != 0 && v9 != -4096 && v9 != -8192 )
                  {
                    sub_BD60C0(v5 + 5);
                    v8 = v5[3];
                  }
                  i = v8;
                }
              }
              *v5 = &unk_49DB368;
              if ( i != -4096 && i != 0 && i != -8192 )
                sub_BD60C0(v5 + 1);
              v5 += 8;
              if ( v6 == v5 )
                break;
            }
            v17 = &unk_49DB368;
            if ( v19 != 0 && v19 != -4096 && v19 != -8192 )
              sub_BD60C0(v18);
          }
          if ( v15 != 0 && v15 != -4096 && v15 != -8192 )
            sub_BD60C0(v14);
        }
        sub_C7D6A0(*(_QWORD *)(v3 + 8), (unsigned __int64)*(unsigned int *)(v3 + 24) << 6, 8);
        j_j___libc_free_0(v3);
      }
    }
    while ( a1 != v2 );
  }
}
