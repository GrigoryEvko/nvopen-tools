// Function: sub_2BB7E40
// Address: 0x2bb7e40
//
void __fastcall sub_2BB7E40(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  unsigned __int64 *v5; // rax
  __int64 v6; // rcx
  __int64 v7; // r9
  unsigned __int64 *v8; // r12
  unsigned __int64 *v9; // r15
  unsigned __int64 *v10; // r8
  __int64 v11; // rdx
  __int64 v12; // rbx
  unsigned __int64 *v13; // rsi
  __int64 v14; // rdi
  __int64 v15; // r8
  int v16; // eax
  __int64 v17; // [rsp+0h] [rbp-40h]

  v3 = 0x1FFFFFFFFFFFFFFLL;
  *a1 = a3;
  if ( a3 <= 0x1FFFFFFFFFFFFFFLL )
    v3 = a3;
  a1[1] = 0;
  a1[2] = 0;
  if ( a3 > 0 )
  {
    while ( 1 )
    {
      v5 = (unsigned __int64 *)sub_2207800(v3 << 6);
      v8 = v5;
      if ( v5 )
        break;
      v3 >>= 1;
      if ( !v3 )
        return;
    }
    v9 = &v5[8 * v3];
    v10 = v5;
    *v5 = (unsigned __int64)(v5 + 2);
    v5[1] = 0x300000000LL;
    v11 = *(unsigned int *)(a2 + 8);
    if ( (_DWORD)v11 )
    {
      sub_2BB7BD0((__int64)v5, (unsigned __int64 *)a2, v11, v6, (__int64)v5, v7);
      v10 = v8;
    }
    v12 = (__int64)(v8 + 8);
    if ( v9 != v8 + 8 )
    {
      do
      {
        while ( 1 )
        {
          v16 = *(_DWORD *)(v12 - 56);
          v11 = v12 + 16;
          v15 = v12 - 64;
          *(_DWORD *)(v12 + 8) = 0;
          *(_QWORD *)v12 = v12 + 16;
          *(_DWORD *)(v12 + 12) = 3;
          if ( !v16 )
            break;
          v13 = (unsigned __int64 *)(v12 - 64);
          v14 = v12;
          v12 += 64;
          v17 = v15;
          sub_2BB7BD0(v14, v13, v11, v6, v15, v7);
          v15 = v17;
          if ( v9 == (unsigned __int64 *)v12 )
            goto LABEL_13;
        }
        v12 += 64;
      }
      while ( v9 != (unsigned __int64 *)v12 );
LABEL_13:
      v10 = (unsigned __int64 *)(v15 + 64);
    }
    sub_2BB7BD0(a2, v10, v11, v6, (__int64)v10, v7);
    a1[2] = (__int64)v8;
    a1[1] = v3;
  }
}
