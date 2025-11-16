// Function: sub_AB49F0
// Address: 0xab49f0
//
__int64 __fastcall sub_AB49F0(__int64 a1, __int64 a2, int a3, unsigned int a4)
{
  __int64 v4; // rbp
  __int64 v5; // r12
  __int64 v6; // r13
  unsigned int v9; // eax
  unsigned int v10; // eax
  unsigned int v11; // eax
  int v12; // eax
  unsigned int v13; // eax
  __int64 v14; // r8
  unsigned __int64 v15; // rdx
  unsigned int v16; // [rsp-90h] [rbp-90h]
  unsigned int v17; // [rsp-90h] [rbp-90h]
  __int64 v18; // [rsp-88h] [rbp-88h] BYREF
  int v19; // [rsp-80h] [rbp-80h]
  unsigned __int64 v20; // [rsp-78h] [rbp-78h] BYREF
  unsigned int v21; // [rsp-70h] [rbp-70h]
  unsigned __int64 v22; // [rsp-68h] [rbp-68h] BYREF
  unsigned int v23; // [rsp-60h] [rbp-60h]
  unsigned __int64 v24; // [rsp-58h] [rbp-58h] BYREF
  unsigned int v25; // [rsp-50h] [rbp-50h]
  __int64 v26; // [rsp-48h] [rbp-48h] BYREF
  int v27; // [rsp-40h] [rbp-40h]
  __int64 v28; // [rsp-28h] [rbp-28h]
  __int64 v29; // [rsp-20h] [rbp-20h]
  __int64 v30; // [rsp-8h] [rbp-8h]

  v30 = v4;
  v29 = v6;
  v28 = v5;
  switch ( a3 )
  {
    case '&':
      sub_AB4490(a1, a2, a4);
      return a1;
    case '\'':
      sub_AB3F90(a1, a2, a4);
      return a1;
    case '(':
      sub_AB41D0(a1, a2, a4);
      return a1;
    case ')':
    case '*':
      if ( a4 != *(_DWORD *)(a2 + 8) )
        goto LABEL_4;
      sub_AAF450(a1, a2);
      return a1;
    case '+':
      v16 = *(_DWORD *)(a2 + 8);
      sub_9691E0((__int64)&v18, v16, 0, 0, 0);
      sub_9691E0((__int64)&v20, v16, -1, 1u, 0);
      if ( a4 <= v16 )
        goto LABEL_13;
      sub_C449B0(&v26, &v18, a4);
      sub_AAD550(&v18, &v26);
      sub_969240(&v26);
      sub_C449B0(&v26, &v20, a4);
      goto LABEL_21;
    case ',':
      v17 = *(_DWORD *)(a2 + 8);
      sub_986680((__int64)&v18, v17);
      v13 = v17;
      v21 = v17;
      v14 = ~(1LL << ((unsigned __int8)v17 - 1));
      if ( v17 <= 0x40 )
      {
        v15 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v17;
        if ( !v17 )
          v15 = 0;
        v20 = v15;
        goto LABEL_18;
      }
      sub_C43690(&v20, -1, 1);
      v13 = v17;
      v14 = ~(1LL << ((unsigned __int8)v17 - 1));
      if ( v21 <= 0x40 )
      {
LABEL_18:
        v20 &= v14;
        goto LABEL_19;
      }
      *(_QWORD *)(v20 + 8LL * ((v17 - 1) >> 6)) &= ~(1LL << ((unsigned __int8)v17 - 1));
LABEL_19:
      if ( a4 > v13 )
      {
        sub_C44830(&v26, &v18, a4);
        sub_AAD550(&v18, &v26);
        sub_969240(&v26);
        sub_C44830(&v26, &v20, a4);
LABEL_21:
        sub_AAD550((__int64 *)&v20, &v26);
        sub_969240(&v26);
      }
LABEL_13:
      v11 = v21;
      v21 = 0;
      v23 = v11;
      v22 = v20;
      sub_C46A40(&v22, 1);
      v25 = v23;
      v23 = 0;
      v24 = v22;
      v12 = v19;
      v19 = 0;
      v27 = v12;
      v26 = v18;
      sub_9875E0(a1, &v26, (__int64 *)&v24);
      sub_969240(&v26);
      sub_969240((__int64 *)&v24);
      sub_969240((__int64 *)&v22);
      sub_969240((__int64 *)&v20);
      sub_969240(&v18);
      return a1;
    case '-':
    case '.':
    case '/':
    case '0':
    case '2':
LABEL_4:
      sub_AADB10(a1, a4, 1);
      return a1;
    case '1':
      v9 = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 8) = v9;
      if ( v9 > 0x40 )
        sub_C43780(a1, a2);
      else
        *(_QWORD *)a1 = *(_QWORD *)a2;
      v10 = *(_DWORD *)(a2 + 24);
      *(_DWORD *)(a1 + 24) = v10;
      if ( v10 > 0x40 )
        sub_C43780(a1 + 16, a2 + 16);
      else
        *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
      return a1;
    default:
      BUG();
  }
}
