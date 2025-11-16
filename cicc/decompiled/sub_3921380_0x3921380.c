// Function: sub_3921380
// Address: 0x3921380
//
void __fastcall sub_3921380(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, unsigned __int64 a5)
{
  __int64 v8; // rbx
  __int64 v9; // rcx
  unsigned __int64 v10; // rdx
  char v11; // si
  char v12; // al
  char *v13; // rax
  __int64 v14; // r12
  __int64 v15; // r14
  __int64 v16; // r12
  __int64 v17; // rdi
  __int64 v18; // r14
  unsigned __int64 v19; // r15
  char v20; // si
  char v21; // al
  char *v22; // rax
  _QWORD *v23; // r15
  __int64 v24; // rax
  unsigned __int64 v25; // [rsp+10h] [rbp-80h]
  __int64 v26; // [rsp+18h] [rbp-78h]
  __int64 v27; // [rsp+18h] [rbp-78h]
  unsigned __int64 v30; // [rsp+38h] [rbp-58h] BYREF
  _QWORD v31[2]; // [rsp+40h] [rbp-50h] BYREF
  __int64 v32; // [rsp+50h] [rbp-40h]
  int v33; // [rsp+58h] [rbp-38h]

  if ( a5 )
  {
    v8 = a4;
    sub_391B370(a1, (__int64)v31, 10);
    v9 = *(_QWORD *)(a1 + 8);
    v10 = a5;
    *(_DWORD *)(a1 + 56) = v33;
    do
    {
      while ( 1 )
      {
        v11 = v10 & 0x7F;
        v12 = v10 & 0x7F | 0x80;
        v10 >>= 7;
        if ( v10 )
          v11 = v12;
        v13 = *(char **)(v9 + 24);
        if ( (unsigned __int64)v13 >= *(_QWORD *)(v9 + 16) )
          break;
        *(_QWORD *)(v9 + 24) = v13 + 1;
        *v13 = v11;
        if ( !v10 )
          goto LABEL_8;
      }
      v25 = v10;
      v26 = v9;
      sub_16E7DE0(v9, v11);
      v10 = v25;
      v9 = v26;
    }
    while ( v25 );
LABEL_8:
    v14 = 16 * a5;
    v27 = a4 + v14;
    if ( a4 != a4 + v14 )
    {
      do
      {
        v15 = *(_QWORD *)(v8 + 8);
        v16 = *(_QWORD *)v15 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v16
          || (v16 = 0, (*(_BYTE *)(v15 + 9) & 0xC) == 8)
          && (*(_BYTE *)(v15 + 8) |= 4u,
              v16 = (__int64)sub_38CE440(*(_QWORD *)(v15 + 24)),
              *(_QWORD *)v15 = v16 | *(_QWORD *)v15 & 7LL,
              v15 = *(_QWORD *)(v8 + 8),
              v16) )
        {
          v16 = *(_QWORD *)(v16 + 24);
        }
        v17 = *(_QWORD *)(v15 + 136);
        v30 = 0;
        if ( !sub_38CF2A0(v17, &v30, a3) )
          sub_16BD130(".size expression must be evaluatable", 1u);
        v18 = *(_QWORD *)(a1 + 8);
        v19 = v30;
        do
        {
          while ( 1 )
          {
            v20 = v19 & 0x7F;
            v21 = v19 & 0x7F | 0x80;
            v19 >>= 7;
            if ( v19 )
              v20 = v21;
            v22 = *(char **)(v18 + 24);
            if ( (unsigned __int64)v22 >= *(_QWORD *)(v18 + 16) )
              break;
            *(_QWORD *)(v18 + 24) = v22 + 1;
            *v22 = v20;
            if ( !v19 )
              goto LABEL_18;
          }
          sub_16E7DE0(v18, v20);
        }
        while ( v19 );
LABEL_18:
        v23 = *(_QWORD **)(a1 + 8);
        v8 += 16;
        v24 = (*(__int64 (__fastcall **)(_QWORD *))(*v23 + 64LL))(v23);
        *(_QWORD *)(v16 + 184) = v24 + v23[3] - v23[1] - v32;
        sub_390B9B0(a2, *(_QWORD **)(a1 + 8), v16, a3);
      }
      while ( v8 != v27 );
    }
    sub_39207C0(
      a1,
      *(_QWORD *)(a1 + 32),
      0xCCCCCCCCCCCCCCCDLL * ((__int64)(*(_QWORD *)(a1 + 40) - *(_QWORD *)(a1 + 32)) >> 3),
      v32);
    sub_3919EA0(a1, v31);
  }
}
