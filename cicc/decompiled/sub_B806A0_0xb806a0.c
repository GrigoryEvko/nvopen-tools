// Function: sub_B806A0
// Address: 0xb806a0
//
__int64 __fastcall sub_B806A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  _QWORD *v5; // r14
  size_t v6; // r15
  unsigned int v7; // eax
  unsigned int v8; // r8d
  _QWORD *v9; // r9
  __int64 v10; // rax
  _QWORD *v11; // rdi
  __int64 v12; // r14
  int v13; // r13d
  __int64 v14; // rdx
  _BYTE *v15; // rsi
  __int64 v16; // rax
  unsigned int v17; // r8d
  _QWORD *v18; // r9
  _QWORD *v19; // rcx
  __int64 *v20; // rdx
  _QWORD *v22; // [rsp+0h] [rbp-90h]
  __int64 v23; // [rsp+20h] [rbp-70h]
  _QWORD *v24; // [rsp+28h] [rbp-68h]
  unsigned int v25; // [rsp+30h] [rbp-60h]
  unsigned int v26; // [rsp+34h] [rbp-5Ch]
  void *src; // [rsp+40h] [rbp-50h] BYREF
  size_t n; // [rsp+48h] [rbp-48h]
  _QWORD v29[8]; // [rsp+50h] [rbp-40h] BYREF

  v3 = *(_QWORD *)(a2 + 32);
  v23 = a2 + 24;
  if ( v3 != a2 + 24 )
  {
    v25 = 0;
    while ( 1 )
    {
      v12 = v3 - 56;
      if ( !v3 )
        v12 = 0;
      v13 = sub_B2BED0(v12);
      v15 = (_BYTE *)sub_BD5D20(v12);
      src = v29;
      if ( v15 )
      {
        sub_B7EB70((__int64 *)&src, v15, (__int64)&v15[v14]);
        v5 = src;
        v6 = n;
      }
      else
      {
        LOBYTE(v29[0]) = 0;
        v6 = 0;
        n = 0;
        v5 = v29;
      }
      v7 = sub_C92610(v5, v6);
      v8 = sub_C92740(a3, v5, v6, v7);
      v9 = (_QWORD *)(*(_QWORD *)a3 + 8LL * v8);
      v10 = *v9;
      if ( !*v9 )
        goto LABEL_14;
      if ( v10 == -8 )
        break;
LABEL_6:
      *(_DWORD *)(v10 + 8) = v13;
      v11 = src;
      *(_DWORD *)(v10 + 12) = 0;
      if ( v11 != v29 )
        j_j___libc_free_0(v11, v29[0] + 1LL);
      v25 += v13;
      v3 = *(_QWORD *)(v3 + 8);
      if ( v23 == v3 )
        return v25;
    }
    --*(_DWORD *)(a3 + 16);
LABEL_14:
    v24 = v9;
    v26 = v8;
    v16 = sub_C7D670(v6 + 17, 8);
    v17 = v26;
    v18 = v24;
    v19 = (_QWORD *)v16;
    if ( v6 )
    {
      v22 = (_QWORD *)v16;
      memcpy((void *)(v16 + 16), v5, v6);
      v17 = v26;
      v18 = v24;
      v19 = v22;
    }
    *((_BYTE *)v19 + v6 + 16) = 0;
    *v19 = v6;
    v19[1] = 0;
    *v18 = v19;
    ++*(_DWORD *)(a3 + 12);
    v20 = (__int64 *)(*(_QWORD *)a3 + 8LL * (unsigned int)sub_C929D0(a3, v17));
    v10 = *v20;
    if ( *v20 )
      goto LABEL_18;
    while ( 1 )
    {
      do
      {
        v10 = v20[1];
        ++v20;
      }
      while ( !v10 );
LABEL_18:
      if ( v10 != -8 )
        goto LABEL_6;
    }
  }
  return 0;
}
