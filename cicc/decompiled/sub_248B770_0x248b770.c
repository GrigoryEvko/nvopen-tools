// Function: sub_248B770
// Address: 0x248b770
//
void __fastcall sub_248B770(unsigned __int64 **a1, unsigned __int64 a2, __int64 a3)
{
  unsigned __int64 *v5; // rbx
  _DWORD *v6; // rax
  _QWORD *v7; // rax
  unsigned __int64 v8; // r14
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // r15
  unsigned __int64 v11; // r13
  _DWORD *v12; // rax
  unsigned __int64 v13; // rsi
  char v14; // al
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // r15
  _QWORD *v17; // rax
  char *v18; // r13
  _QWORD *v19; // r9
  _QWORD *v20; // rsi
  unsigned __int64 v21; // rdi
  _QWORD *v22; // rcx
  unsigned __int64 v23; // rdx
  char *v24; // rax
  __int64 n; // [rsp+8h] [rbp-48h]
  unsigned __int64 v26[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = *a1;
  v26[0] = a2;
  v6 = sub_248B700(v5, a2 % v5[1], v26, a2);
  if ( !v6 || !*(_QWORD *)v6 )
  {
    v7 = (_QWORD *)sub_22077B0(0x20u);
    v8 = (unsigned __int64)v7;
    if ( v7 )
      *v7 = 0;
    v9 = v26[0];
    *(_QWORD *)(v8 + 16) = a3;
    *(_QWORD *)(v8 + 8) = v9;
    v10 = v5[1];
    v11 = a2 % v10;
    v12 = sub_248B700(v5, a2 % v10, (_DWORD *)(v8 + 8), a2);
    if ( v12 && *(_QWORD *)v12 )
    {
      j_j___libc_free_0(v8);
    }
    else
    {
      v13 = v10;
      v14 = sub_222DA10((__int64)(v5 + 4), v10, v5[3], 1);
      v16 = v15;
      if ( v14 )
      {
        if ( v15 == 1 )
        {
          v18 = (char *)(v5 + 6);
          v5[6] = 0;
          v19 = v5 + 6;
        }
        else
        {
          if ( v15 > 0xFFFFFFFFFFFFFFFLL )
            sub_4261EA(v5 + 4, v13, v15);
          n = 8 * v15;
          v18 = (char *)sub_22077B0(8 * v15);
          memset(v18, 0, n);
          v19 = v5 + 6;
        }
        v20 = (_QWORD *)v5[2];
        v5[2] = 0;
        if ( v20 )
        {
          v21 = 0;
          do
          {
            v22 = v20;
            v20 = (_QWORD *)*v20;
            v23 = v22[3] % v16;
            v24 = &v18[8 * v23];
            if ( *(_QWORD *)v24 )
            {
              *v22 = **(_QWORD **)v24;
              **(_QWORD **)v24 = v22;
            }
            else
            {
              *v22 = v5[2];
              v5[2] = (unsigned __int64)v22;
              *(_QWORD *)v24 = v5 + 2;
              if ( *v22 )
                *(_QWORD *)&v18[8 * v21] = v22;
              v21 = v23;
            }
          }
          while ( v20 );
        }
        if ( (_QWORD *)*v5 != v19 )
          j_j___libc_free_0(*v5);
        *v5 = (unsigned __int64)v18;
        v5[1] = v16;
        v11 = a2 % v16;
      }
      *(_QWORD *)(v8 + 24) = a2;
      v17 = *(_QWORD **)(*v5 + 8 * v11);
      if ( v17 )
      {
        *(_QWORD *)v8 = *v17;
        **(_QWORD **)(*v5 + 8 * v11) = v8;
      }
      else
      {
        *(_QWORD *)v8 = v5[2];
        v5[2] = v8;
        if ( *(_QWORD *)v8 )
          *(_QWORD *)(*v5 + 8 * (*(_QWORD *)(*(_QWORD *)v8 + 24LL) % v5[1])) = v8;
        *(_QWORD *)(*v5 + 8 * v11) = v5 + 2;
      }
      ++v5[3];
    }
  }
}
