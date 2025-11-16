// Function: sub_26C21F0
// Address: 0x26c21f0
//
void __fastcall sub_26C21F0(__int64 a1, __int64 a2, unsigned __int8 *a3, char a4)
{
  _QWORD *v4; // rbx
  __int64 v5; // r13
  char *v6; // r14
  _QWORD *v7; // rsi
  __int64 v8; // r12
  size_t v9; // rax
  __int64 v10; // r12
  __int64 v11; // r12
  unsigned __int64 *v12; // r12
  __int64 v13; // r8
  unsigned __int64 *v14; // r15
  unsigned __int64 v15; // rdi
  __int64 *v18; // [rsp+18h] [rbp-2A8h]
  __int64 v19; // [rsp+20h] [rbp-2A0h]
  __m128i v20; // [rsp+30h] [rbp-290h] BYREF
  unsigned __int64 v21[2]; // [rsp+40h] [rbp-280h] BYREF
  __int64 v22; // [rsp+50h] [rbp-270h] BYREF
  __int64 *v23; // [rsp+60h] [rbp-260h]
  __int64 v24; // [rsp+70h] [rbp-250h] BYREF
  unsigned __int64 v25[2]; // [rsp+90h] [rbp-230h] BYREF
  __int64 v26; // [rsp+A0h] [rbp-220h] BYREF
  __int64 *v27; // [rsp+B0h] [rbp-210h]
  __int64 v28; // [rsp+C0h] [rbp-200h] BYREF
  _QWORD v29[10]; // [rsp+E0h] [rbp-1E0h] BYREF
  unsigned __int64 *v30; // [rsp+130h] [rbp-190h]
  unsigned int v31; // [rsp+138h] [rbp-188h]
  char v32; // [rsp+140h] [rbp-180h] BYREF

  v4 = *(_QWORD **)a2;
  v5 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v5 )
  {
    v6 = "hotness: '";
    if ( !a4 )
      v6 = "size: '";
    do
    {
      while ( 1 )
      {
        v7 = (_QWORD *)*v4;
        v8 = *(_QWORD *)(*v4 - 32LL);
        if ( v8 )
        {
          if ( !*(_BYTE *)v8 && *(_QWORD *)(v8 + 24) == v7[10] )
          {
            v19 = v7[5];
            v18 = *(__int64 **)(a1 + 1288);
            sub_B157E0((__int64)&v20, v7 + 6);
            sub_B17850((__int64)v29, *(_QWORD *)(a1 + 1528), (__int64)"InlineAttempt", 13, &v20, v19);
            sub_B18290((__int64)v29, "previous inlining reattempted for ", 0x22u);
            v9 = strlen(v6);
            sub_B18290((__int64)v29, v6, v9);
            sub_B16080((__int64)v21, "Callee", 6, (unsigned __int8 *)v8);
            v10 = sub_B826F0((__int64)v29, (__int64)v21);
            sub_B18290(v10, "' into '", 8u);
            sub_B16080((__int64)v25, "Caller", 6, a3);
            v11 = sub_B826F0(v10, (__int64)v25);
            sub_B18290(v11, "'", 1u);
            sub_1049740(v18, v11);
            if ( v27 != &v28 )
              j_j___libc_free_0((unsigned __int64)v27);
            if ( (__int64 *)v25[0] != &v26 )
              j_j___libc_free_0(v25[0]);
            if ( v23 != &v24 )
              j_j___libc_free_0((unsigned __int64)v23);
            if ( (__int64 *)v21[0] != &v22 )
              j_j___libc_free_0(v21[0]);
            v12 = v30;
            v29[0] = &unk_49D9D40;
            v13 = 10LL * v31;
            v14 = &v30[v13];
            if ( v30 != &v30[v13] )
            {
              do
              {
                v14 -= 10;
                v15 = v14[4];
                if ( (unsigned __int64 *)v15 != v14 + 6 )
                  j_j___libc_free_0(v15);
                if ( (unsigned __int64 *)*v14 != v14 + 2 )
                  j_j___libc_free_0(*v14);
              }
              while ( v12 != v14 );
              v14 = v30;
            }
            if ( v14 != (unsigned __int64 *)&v32 )
              break;
          }
        }
        if ( (_QWORD *)v5 == ++v4 )
          return;
      }
      ++v4;
      _libc_free((unsigned __int64)v14);
    }
    while ( (_QWORD *)v5 != v4 );
  }
}
