// Function: sub_1DC26F0
// Address: 0x1dc26f0
//
__int64 __fastcall sub_1DC26F0(__int64 *a1, __int64 a2)
{
  unsigned __int16 *v3; // rbx
  __int64 result; // rax
  int v5; // r8d
  int v6; // r9d
  unsigned __int16 *v7; // rcx
  _QWORD *v8; // rax
  int v9; // r15d
  unsigned __int16 v10; // r14
  __int64 v11; // rdx
  _WORD *v12; // rbx
  __int64 v13; // rdi
  unsigned __int16 *v14; // r12
  unsigned __int16 v15; // r14
  unsigned __int16 *v16; // rbx
  __int64 v17; // rdx
  unsigned __int16 *v18; // [rsp+0h] [rbp-40h]
  unsigned __int16 *v19; // [rsp+8h] [rbp-38h]
  unsigned __int16 *v20; // [rsp+8h] [rbp-38h]

  v3 = *(unsigned __int16 **)(a2 + 160);
  v18 = v3;
  result = sub_1DD77D0(a2);
  if ( v3 != (unsigned __int16 *)result )
  {
    v7 = (unsigned __int16 *)result;
    do
    {
      v8 = (_QWORD *)*a1;
      if ( !*a1 )
        BUG();
      v9 = *((_DWORD *)v7 + 1);
      v10 = *v7;
      v11 = v8[1] + 24LL * *v7;
      v12 = (_WORD *)(v8[7] + 2LL * *(unsigned int *)(v11 + 4));
      if ( !*v12
        || (v13 = *(unsigned int *)(v11 + 12), v11 = v8[11], v14 = (unsigned __int16 *)(v11 + 2 * v13), v9 == -1) )
      {
        v19 = v7;
        result = sub_1DC1BF0(a1, v10, v11, (__int64)v7, v5, v6);
        v7 = v19;
      }
      else
      {
        v15 = *v12 + v10;
        v16 = v12 + 1;
        while ( 1 )
        {
          v17 = *v14;
          if ( (*(_DWORD *)(v8[31] + 4 * v17) & v9) != 0 )
          {
            v20 = v7;
            sub_1DC1BF0(a1, v15, v17, (__int64)v7, v5, v6);
            v7 = v20;
          }
          result = *v16;
          ++v14;
          ++v16;
          if ( !(_WORD)result )
            break;
          v15 += result;
          v8 = (_QWORD *)*a1;
        }
      }
      v7 += 4;
    }
    while ( v18 != v7 );
  }
  return result;
}
