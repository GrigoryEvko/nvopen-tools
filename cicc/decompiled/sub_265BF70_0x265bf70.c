// Function: sub_265BF70
// Address: 0x265bf70
//
unsigned __int64 *__fastcall sub_265BF70(unsigned __int64 *a1, __int64 a2, unsigned __int64 *a3)
{
  unsigned __int64 *v3; // rbx
  unsigned __int64 *v4; // r12
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // r15
  __int64 v7; // r9
  __int64 v8; // r9
  _QWORD *v9; // rdx
  __int64 v10; // rax
  _QWORD *v11; // r14
  _QWORD *v12; // r13
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v17; // rsi
  _QWORD *v18; // r14
  _QWORD *v19; // r13
  __int64 v20; // rsi
  __int64 v21; // [rsp+8h] [rbp-B8h]
  __int64 v23; // [rsp+28h] [rbp-98h]
  _QWORD v24[2]; // [rsp+38h] [rbp-88h] BYREF
  __int64 v25; // [rsp+48h] [rbp-78h]
  __int64 v26; // [rsp+50h] [rbp-70h]
  void *v27; // [rsp+60h] [rbp-60h]
  _QWORD v28[2]; // [rsp+68h] [rbp-58h] BYREF
  __int64 v29; // [rsp+78h] [rbp-48h]
  __int64 v30; // [rsp+80h] [rbp-40h]

  v21 = a2 - (_QWORD)a1;
  v23 = (a2 - (__int64)a1) >> 3;
  if ( a2 - (__int64)a1 <= 0 )
    return a3;
  v3 = a1;
  v4 = a3;
  do
  {
    v5 = *v3;
    *v3 = 0;
    v6 = *v4;
    *v4 = v5;
    if ( v6 )
    {
      if ( *(_BYTE *)(v6 + 64) )
      {
        v17 = *(unsigned int *)(v6 + 56);
        *(_BYTE *)(v6 + 64) = 0;
        if ( (_DWORD)v17 )
        {
          v18 = *(_QWORD **)(v6 + 40);
          v19 = &v18[2 * v17];
          do
          {
            if ( *v18 != -8192 && *v18 != -4096 )
            {
              v20 = v18[1];
              if ( v20 )
                sub_B91220((__int64)(v18 + 1), v20);
            }
            v18 += 2;
          }
          while ( v19 != v18 );
          v17 = *(unsigned int *)(v6 + 56);
        }
        sub_C7D6A0(*(_QWORD *)(v6 + 40), 16 * v17, 8);
      }
      if ( *(_DWORD *)(v6 + 24) )
      {
        v25 = -4096;
        v26 = 0;
        v29 = -8192;
        v30 = 0;
        v7 = *(unsigned int *)(v6 + 24);
        v24[0] = 2;
        v24[1] = 0;
        v8 = v7 << 6;
        v28[0] = 2;
        v28[1] = 0;
        v9 = *(_QWORD **)(v6 + 8);
        v27 = &unk_49DD7B0;
        if ( v9 != (_QWORD *)((char *)v9 + v8) )
        {
          v10 = -4096;
          v11 = (_QWORD *)((char *)v9 + v8);
          v12 = v9;
          while ( 1 )
          {
            v13 = v12[3];
            if ( v13 != v10 )
            {
              v10 = v29;
              if ( v13 != v29 )
              {
                v14 = v12[7];
                if ( v14 != -4096 && v14 != 0 && v14 != -8192 )
                {
                  sub_BD60C0(v12 + 5);
                  v13 = v12[3];
                }
                v10 = v13;
              }
            }
            *v12 = &unk_49DB368;
            if ( v10 != -4096 && v10 != 0 && v10 != -8192 )
              sub_BD60C0(v12 + 1);
            v12 += 8;
            if ( v11 == v12 )
              break;
            v10 = v25;
          }
          v27 = &unk_49DB368;
          if ( v29 != 0 && v29 != -4096 && v29 != -8192 )
            sub_BD60C0(v28);
        }
        if ( v25 != 0 && v25 != -4096 && v25 != -8192 )
          sub_BD60C0(v24);
      }
      sub_C7D6A0(*(_QWORD *)(v6 + 8), (unsigned __int64)*(unsigned int *)(v6 + 24) << 6, 8);
      j_j___libc_free_0(v6);
    }
    ++v3;
    ++v4;
    --v23;
  }
  while ( v23 );
  v15 = v21;
  if ( v21 <= 0 )
    v15 = 8;
  return (unsigned __int64 *)((char *)a3 + v15);
}
