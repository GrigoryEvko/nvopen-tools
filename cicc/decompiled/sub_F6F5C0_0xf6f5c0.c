// Function: sub_F6F5C0
// Address: 0xf6f5c0
//
__int64 __fastcall sub_F6F5C0(__int64 a1, __int64 a2, __int64 a3, int a4, int a5)
{
  __int64 v7; // r13
  _QWORD *v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // r15
  _QWORD *v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rbx
  __int64 v16; // r14
  __int64 v17; // rdx
  unsigned int v18; // esi
  __int64 v19; // [rsp+8h] [rbp-B8h]
  __int64 v21; // [rsp+18h] [rbp-A8h]
  unsigned int v23; // [rsp+24h] [rbp-9Ch]
  int v25[8]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v26; // [rsp+50h] [rbp-70h]
  _QWORD v27[4]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v28; // [rsp+80h] [rbp-40h]

  if ( *(_DWORD *)(*(_QWORD *)(a3 + 8) + 32LL) )
  {
    v21 = *(unsigned int *)(*(_QWORD *)(a3 + 8) + 32LL);
    v7 = 0;
    v23 = a4 - 53;
    do
    {
      while ( 1 )
      {
        v8 = *(_QWORD **)(a1 + 72);
        v26 = 257;
        v9 = sub_BCB2D0(v8);
        v10 = sub_ACD640(v9, v7, 0);
        v11 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 80) + 96LL))(
                *(_QWORD *)(a1 + 80),
                a3,
                v10);
        if ( !v11 )
        {
          v28 = 257;
          v13 = sub_BD2C40(72, 2u);
          v11 = (__int64)v13;
          if ( v13 )
            sub_B4DE80((__int64)v13, a3, v10, (__int64)v27, 0, 0);
          (*(void (__fastcall **)(_QWORD, __int64, int *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
            *(_QWORD *)(a1 + 88),
            v11,
            v25,
            *(_QWORD *)(a1 + 56),
            *(_QWORD *)(a1 + 64));
          v14 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
          if ( *(_QWORD *)a1 != v14 )
          {
            v19 = a1;
            v15 = *(_QWORD *)a1;
            v16 = v14;
            do
            {
              v17 = *(_QWORD *)(v15 + 8);
              v18 = *(_DWORD *)v15;
              v15 += 16;
              sub_B99FD0(v11, v18, v17);
            }
            while ( v16 != v15 );
            a1 = v19;
          }
        }
        if ( v23 <= 1 )
          break;
        v27[0] = "bin.rdx";
        ++v7;
        v28 = 259;
        a2 = sub_F6BB60((__int64 *)a1, a4, a2, v11, v25[0], 0, (__int64)v27, 0);
        if ( v21 == v7 )
          return a2;
      }
      ++v7;
      a2 = sub_F6F180(a1, a5, a2, v11);
    }
    while ( v21 != v7 );
  }
  return a2;
}
