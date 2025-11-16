// Function: sub_1C014D0
// Address: 0x1c014d0
//
__int64 __fastcall sub_1C014D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  unsigned int v6; // eax
  __int64 v7; // rax
  __int64 v8; // r15
  unsigned int v9; // r12d
  unsigned int v10; // eax
  __int64 v11; // rcx
  unsigned int v12; // edx
  _QWORD *v13; // rax
  int v14; // r13d
  __int64 v15; // rbx
  __int64 v16; // r14
  __int64 v18; // rax
  _QWORD *v19; // rax
  __int64 v20; // rdx
  _QWORD *v21; // rbx
  __int64 v22; // r13
  void *v23; // rax
  const void *v24; // rsi
  _QWORD *v25; // [rsp+8h] [rbp-58h]
  __int64 v26; // [rsp+10h] [rbp-50h] BYREF
  void *v27; // [rsp+18h] [rbp-48h]
  __int64 v28; // [rsp+20h] [rbp-40h]
  unsigned int v29; // [rsp+28h] [rbp-38h]

  v4 = a2 + 40;
  if ( *(_QWORD *)(a2 + 48) == a2 + 40 )
  {
    v18 = *(_QWORD *)(sub_1BFDF20(a1, a2) + 8);
    v9 = (unsigned int)(*(_DWORD *)(v18 + 16) + 63) >> 6;
    if ( v9 )
    {
      v19 = *(_QWORD **)v18;
      v20 = v9 - 1;
      v9 = 0;
      v21 = v19 + 1;
      v22 = (__int64)&v19[v20 + 1];
      while ( 1 )
      {
        v9 += sub_39FAC40(*v19);
        v19 = v21;
        if ( (_QWORD *)v22 == v21 )
          break;
        ++v21;
      }
    }
  }
  else
  {
    v26 = 0;
    v27 = 0;
    v28 = 0;
    v29 = 0;
    if ( a3 )
    {
      j___libc_free_0(0);
      v6 = *(_DWORD *)(a3 + 24);
      v29 = v6;
      if ( v6 )
      {
        v23 = (void *)sub_22077B0(16LL * v6);
        v24 = *(const void **)(a3 + 8);
        v27 = v23;
        v28 = *(_QWORD *)(a3 + 16);
        memcpy(v23, v24, 16LL * v29);
      }
      else
      {
        v27 = 0;
        v28 = 0;
      }
    }
    v7 = sub_157ED20(a2);
    if ( !v7 )
    {
      sub_1BFDF20(a1, a2);
LABEL_27:
      BUG();
    }
    v8 = v7 + 24;
    v9 = 0;
    v25 = *(_QWORD **)(sub_1BFDF20(a1, a2) + 8);
    if ( v4 != v8 )
    {
      while ( 1 )
      {
        if ( (*(_DWORD *)(v8 - 4) & 0xFFFFFFF) != 0 )
        {
          v10 = sub_1C009B0(a1, v8 - 24, (__int64)&v26, v25);
          if ( v9 < v10 )
            v9 = v10;
        }
        v8 = *(_QWORD *)(v8 + 8);
        if ( v4 == v8 )
          break;
        if ( !v8 )
          goto LABEL_27;
      }
    }
    v11 = *(_QWORD *)(sub_1BFDF20(a1, a2) + 8);
    v12 = (unsigned int)(*(_DWORD *)(v11 + 16) + 63) >> 6;
    if ( v12 )
    {
      v13 = *(_QWORD **)v11;
      v14 = 0;
      v15 = *(_QWORD *)v11 + 8LL;
      v16 = v15 + 8LL * (v12 - 1);
      while ( 1 )
      {
        v14 += sub_39FAC40(*v13);
        v13 = (_QWORD *)v15;
        if ( v15 == v16 )
          break;
        v15 += 8;
      }
      v9 += v14;
    }
    j___libc_free_0(v27);
  }
  return v9;
}
