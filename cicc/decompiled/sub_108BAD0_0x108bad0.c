// Function: sub_108BAD0
// Address: 0x108bad0
//
__int64 *__fastcall sub_108BAD0(__int64 a1)
{
  __int64 *v1; // rax
  __int64 *v2; // r12
  _QWORD *v3; // rax
  __int64 *v4; // rax
  __int64 *result; // rax
  _QWORD *v6; // r14
  __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rbx
  __int64 v10; // r8
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 *v14; // r15
  unsigned __int64 v15; // rbx
  __int64 v16; // rdi
  _QWORD *v17; // [rsp+8h] [rbp-A8h]
  __int64 *v18; // [rsp+18h] [rbp-98h]
  __int64 *v19; // [rsp+20h] [rbp-90h]
  __int64 *v20; // [rsp+28h] [rbp-88h]
  __int64 v21; // [rsp+30h] [rbp-80h]
  __int64 v22; // [rsp+38h] [rbp-78h]
  __int64 v23[4]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v24[10]; // [rsp+60h] [rbp-50h] BYREF

  *(_WORD *)(a1 + 56) = -3;
  v1 = *(__int64 **)(a1 + 96);
  *(_QWORD *)(a1 + 16) = 0;
  v2 = *(__int64 **)(a1 + 80);
  v19 = v1;
  v3 = *(_QWORD **)(a1 + 104);
  *(_QWORD *)(a1 + 24) = 0;
  v17 = v3;
  v4 = *(__int64 **)(a1 + 112);
  *(_QWORD *)(a1 + 32) = 0;
  v18 = v4;
  result = v24;
  *(_QWORD *)(a1 + 40) = 0;
  *(_DWORD *)(a1 + 48) = 0;
  while ( v18 != v2 )
  {
    while ( 1 )
    {
      v6 = (_QWORD *)*v2;
      v7 = *(_QWORD *)(*v2 + 48);
      v8 = *(_QWORD *)(*v2 + 16);
      v9 = *(_QWORD *)(*v2 + 24);
      v10 = *(_QWORD *)(*v2 + 32);
      v11 = *(_QWORD *)(*v2 + 40);
      v24[1] = *(_QWORD *)(*v2 + 56);
      v12 = v6[9];
      v13 = v6[8];
      v24[0] = v7;
      v21 = v9;
      v14 = (__int64 *)(v11 + 8);
      v24[3] = v12;
      v23[1] = v9;
      v22 = v8;
      v20 = (__int64 *)v10;
      v24[2] = v13;
      v23[0] = v8;
      v23[2] = v10;
      v23[3] = v11;
      sub_108B970(v23, v24);
      v15 = v6[9] + 8LL;
      if ( v15 > v11 + 8 )
      {
        do
        {
          v16 = *v14++;
          j_j___libc_free_0(v16, 480);
        }
        while ( v15 > (unsigned __int64)v14 );
      }
      v6[9] = v11;
      ++v2;
      v6[6] = v22;
      v6[7] = v21;
      result = v20;
      v6[8] = v20;
      if ( v19 != v2 )
        break;
      v2 = (__int64 *)*++v17;
      result = (__int64 *)(*v17 + 512LL);
      v19 = result;
      if ( v18 == (__int64 *)*v17 )
        return result;
    }
  }
  return result;
}
