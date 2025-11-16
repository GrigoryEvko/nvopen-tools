// Function: sub_2B1ACD0
// Address: 0x2b1acd0
//
char __fastcall sub_2B1ACD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, signed int a5)
{
  __int64 v6; // rdx
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // r15
  _DWORD *v11; // rdi
  _DWORD *v12; // rax
  int *v13; // r8
  _DWORD *v14; // rcx
  _DWORD *v15; // rdi
  _DWORD *v16; // rax
  int *v17; // r8
  _DWORD *v18; // rcx
  _DWORD *v19; // rdi
  _DWORD *v20; // rax
  int *v21; // r8
  _DWORD *v22; // rcx
  __int64 v23; // rbx
  _DWORD *v24; // rax
  int *v25; // r8
  _DWORD *v26; // rcx
  __int64 v28; // r15
  __int64 v29; // rcx
  _DWORD *v30; // rsi
  __int64 v31; // rcx
  int *v32; // r8
  _DWORD *v33; // rax
  int *v34; // r8
  _DWORD *v35; // rcx
  _DWORD *v36; // rsi
  int *v37; // r8
  char v38; // al
  char v39; // al
  __int64 v41; // [rsp+8h] [rbp-48h]
  __int64 v42; // [rsp+10h] [rbp-40h]
  __int64 v43; // [rsp+18h] [rbp-38h]
  __int64 v44; // [rsp+18h] [rbp-38h]
  __int64 v45; // [rsp+18h] [rbp-38h]

  v6 = a2 - a1;
  v8 = a1;
  v9 = (a2 - a1) >> 2;
  if ( v9 > 0 )
  {
    v43 = a5;
    v10 = a5;
    v41 = a1 + 4 * v9;
    while ( 1 )
    {
      v24 = sub_2B094B0((_DWORD *)(a3 + 4LL * (int)v8 * a5), a3 + 4LL * (int)v8 * a5 + v10 * 4);
      if ( v26 != v24 && !(unsigned __int8)sub_B4ED80(v25, v43, a5) )
        return a2 == v8;
      v42 = v8 + 1;
      v11 = (_DWORD *)(a3 + 4LL * a5 * ((int)v8 + 1));
      v12 = sub_2B094B0(v11, (__int64)&v11[v10]);
      if ( v14 != v12 && !(unsigned __int8)sub_B4ED80(v13, v43, a5) )
        return v42 == a2;
      v42 = v8 + 2;
      v15 = (_DWORD *)(a3 + 4LL * a5 * ((int)v8 + 2));
      v16 = sub_2B094B0(v15, (__int64)&v15[v10]);
      if ( v18 != v16 && !(unsigned __int8)sub_B4ED80(v17, v43, a5) )
        return v42 == a2;
      v42 = v8 + 3;
      v19 = (_DWORD *)(a3 + 4LL * a5 * ((int)v8 + 3));
      v20 = sub_2B094B0(v19, (__int64)&v19[v10]);
      if ( v22 == v20 )
      {
        v23 = v8 + 4;
        v8 = v23;
        if ( v23 == v41 )
          goto LABEL_16;
      }
      else
      {
        if ( !(unsigned __int8)sub_B4ED80(v21, v43, a5) )
          return v42 == a2;
        v23 = v8 + 4;
        v8 = v23;
        if ( v23 == v41 )
        {
LABEL_16:
          v6 = a2 - v23;
          goto LABEL_17;
        }
      }
    }
  }
  v23 = a1;
LABEL_17:
  switch ( v6 )
  {
    case 2LL:
      v28 = a5;
      v29 = 4LL * a5;
LABEL_22:
      v30 = (_DWORD *)(a3 + 4LL * (int)v8 * a5 + v29);
      if ( v30 != sub_2B094B0((_DWORD *)(a3 + 4LL * (int)v8 * a5), (__int64)v30) )
      {
        v45 = v31;
        v39 = sub_B4ED80(v32, v28, a5);
        v31 = v45;
        if ( !v39 )
          return a2 == v8;
      }
      ++v8;
LABEL_24:
      v33 = sub_2B094B0((_DWORD *)(a3 + 4LL * (int)v8 * a5), a3 + 4LL * (int)v8 * a5 + v31);
      if ( v35 == v33 || (unsigned __int8)sub_B4ED80(v34, v28, a5) )
        return 1;
      return a2 == v8;
    case 3LL:
      v28 = a5;
      v36 = (_DWORD *)(a3 + 4LL * (int)v8 * a5 + 4LL * a5);
      if ( v36 != sub_2B094B0((_DWORD *)(a3 + 4LL * (int)v8 * a5), (__int64)v36) )
      {
        v44 = v29;
        v38 = sub_B4ED80(v37, a5, a5);
        v29 = v44;
        if ( !v38 )
          return a2 == v8;
      }
      v8 = v23 + 1;
      goto LABEL_22;
    case 1LL:
      v28 = a5;
      v31 = 4LL * a5;
      goto LABEL_24;
  }
  return 1;
}
