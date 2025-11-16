// Function: sub_1AD41A0
// Address: 0x1ad41a0
//
void __fastcall sub_1AD41A0(__int64 a1, __int64 a2)
{
  char v2; // r13
  __int64 v3; // rax
  unsigned __int64 v4; // r12
  __int64 v5; // rdx
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 *v8; // rax
  __int64 v9; // r15
  unsigned int *v10; // rbx
  unsigned int *v11; // r14
  unsigned int v12; // r15d
  __int64 v13; // rax
  unsigned __int8 *v14; // rsi
  __int64 v15; // rax
  _QWORD *v16; // r15
  _QWORD *v17; // rbx
  _QWORD *v18; // r12
  __int64 v19; // r13
  __int64 v20; // rdi
  __int64 v21; // [rsp+10h] [rbp-110h]
  __int64 v22; // [rsp+20h] [rbp-100h]
  __int64 v23; // [rsp+28h] [rbp-F8h]
  char i; // [rsp+30h] [rbp-F0h]
  unsigned __int8 *v25; // [rsp+48h] [rbp-D8h] BYREF
  __int64 v26[5]; // [rsp+50h] [rbp-D0h] BYREF
  int v27; // [rsp+78h] [rbp-A8h]
  __int64 v28; // [rsp+80h] [rbp-A0h]
  __int64 v29; // [rsp+88h] [rbp-98h]
  unsigned __int64 v30[2]; // [rsp+A0h] [rbp-80h] BYREF
  char v31; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v32; // [rsp+B8h] [rbp-68h]
  _QWORD *v33; // [rsp+C0h] [rbp-60h]
  __int64 v34; // [rsp+C8h] [rbp-58h]
  unsigned int v35; // [rsp+D0h] [rbp-50h]
  __int64 v36; // [rsp+E0h] [rbp-40h]
  char v37; // [rsp+E8h] [rbp-38h]
  int v38; // [rsp+ECh] [rbp-34h]

  v2 = byte_4FB6400;
  if ( byte_4FB6400 )
  {
    v3 = *(_QWORD *)(a2 + 8);
    if ( v3 )
    {
      v4 = a1 & 0xFFFFFFFFFFFFFFF8LL;
      v5 = *(_QWORD *)((a1 & 0xFFFFFFFFFFFFFFF8LL) + 40);
      v6 = *(_QWORD *)(v5 + 56);
      if ( !*(_QWORD *)(v3 + 16) )
        sub_4263D6(a1, v6, v5);
      v22 = (*(__int64 (__fastcall **)(__int64, __int64))(v3 + 24))(v3, v6);
      v7 = sub_1632FA0(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v4 + 40) + 56LL) + 40LL));
      v32 = 0;
      v23 = v7;
      v30[0] = (unsigned __int64)&v31;
      v30[1] = 0x100000000LL;
      v8 = (__int64 *)(v4 - 72);
      if ( (a1 & 4) != 0 )
        v8 = (__int64 *)(v4 - 24);
      v33 = 0;
      v34 = 0;
      v9 = *v8;
      v35 = 0;
      v36 = 0;
      v37 = 0;
      v38 = 0;
      if ( *(_BYTE *)(v9 + 16) )
        BUG();
      if ( (*(_BYTE *)(v9 + 18) & 1) != 0 )
      {
        sub_15E08E0(v9, v6);
        v10 = *(unsigned int **)(v9 + 88);
        v11 = &v10[10 * *(_QWORD *)(v9 + 96)];
        if ( (*(_BYTE *)(v9 + 18) & 1) != 0 )
        {
          sub_15E08E0(v9, v6);
          v10 = *(unsigned int **)(v9 + 88);
        }
      }
      else
      {
        v10 = *(unsigned int **)(v9 + 88);
        v11 = &v10[10 * *(_QWORD *)(v9 + 96)];
      }
      for ( i = 0; v11 != v10; v10 += 10 )
      {
        if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) == 15 )
        {
          v12 = sub_15E0370((__int64)v10);
          if ( v12 )
          {
            if ( !(unsigned __int8)sub_15E0300((__int64)v10) && !sub_1648CD0((__int64)v10, 0) )
            {
              if ( !i )
              {
                v36 = *(_QWORD *)(*(_QWORD *)(v4 + 40) + 56LL);
                sub_15D3930((__int64)v30);
              }
              v21 = *(_QWORD *)(v4 + 24 * (v10[8] - (unsigned __int64)(*(_DWORD *)(v4 + 20) & 0xFFFFFFF)));
              i = v2;
              if ( v12 > (unsigned int)sub_1AE99B0(v21, 0, v23, v4, v22, v30) )
              {
                v13 = sub_16498A0(v4);
                v14 = *(unsigned __int8 **)(v4 + 48);
                v26[0] = 0;
                v26[3] = v13;
                v15 = *(_QWORD *)(v4 + 40);
                v26[4] = 0;
                v26[1] = v15;
                v27 = 0;
                v28 = 0;
                v29 = 0;
                v26[2] = v4 + 24;
                v25 = v14;
                if ( v14 )
                {
                  sub_1623A60((__int64)&v25, (__int64)v14, 2);
                  if ( v26[0] )
                    sub_161E7C0((__int64)v26, v26[0]);
                  v26[0] = (__int64)v25;
                  if ( v25 )
                    sub_1623210((__int64)&v25, v25, (__int64)v26);
                }
                v16 = sub_15E8840(v26, v23, v21, v12, 0);
                sub_17CD270(v26);
                sub_14CE830(v22, (__int64)v16);
                i = v2;
              }
            }
          }
        }
      }
      if ( v35 )
      {
        v17 = v33;
        v18 = &v33[2 * v35];
        do
        {
          if ( *v17 != -16 && *v17 != -8 )
          {
            v19 = v17[1];
            if ( v19 )
            {
              v20 = *(_QWORD *)(v19 + 24);
              if ( v20 )
                j_j___libc_free_0(v20, *(_QWORD *)(v19 + 40) - v20);
              j_j___libc_free_0(v19, 56);
            }
          }
          v17 += 2;
        }
        while ( v18 != v17 );
      }
      j___libc_free_0(v33);
      if ( (char *)v30[0] != &v31 )
        _libc_free(v30[0]);
    }
  }
}
