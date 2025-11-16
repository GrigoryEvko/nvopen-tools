// Function: sub_2EBF200
// Address: 0x2ebf200
//
void __fastcall sub_2EBF200(_QWORD *a1)
{
  __int64 v2; // rax
  __int64 v3; // r8
  __int64 v4; // r9
  _BYTE *v5; // rax
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // rax
  int v8; // r13d
  __int64 v9; // r14
  _BYTE *v10; // rsi
  _BYTE *v11; // rdi
  __int64 v12; // rdx
  int v13; // eax
  unsigned __int64 v14; // rdi
  _BYTE *v15; // [rsp+0h] [rbp-70h] BYREF
  __int64 v16; // [rsp+8h] [rbp-68h]
  _BYTE src[48]; // [rsp+10h] [rbp-60h] BYREF
  int v18; // [rsp+40h] [rbp-30h]

  v2 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*a1 + 16LL) + 200LL))(*(_QWORD *)(*a1 + 16LL));
  (*(void (__fastcall **)(_BYTE **, __int64, _QWORD))(*(_QWORD *)v2 + 136LL))(&v15, v2, *a1);
  v5 = v15;
  if ( v15 == src )
  {
    v6 = (unsigned int)v16;
    v7 = *((unsigned int *)a1 + 98);
    v8 = v16;
    if ( (unsigned int)v16 <= v7 )
    {
      v11 = src;
      if ( (_DWORD)v16 )
      {
        memmove((void *)a1[48], src, 8LL * (unsigned int)v16);
        v11 = v15;
      }
    }
    else
    {
      if ( (unsigned int)v16 > (unsigned __int64)*((unsigned int *)a1 + 99) )
      {
        *((_DWORD *)a1 + 98) = 0;
        sub_C8D5F0((__int64)(a1 + 48), a1 + 50, v6, 8u, v3, v4);
        v11 = v15;
        v6 = (unsigned int)v16;
        v7 = 0;
        v10 = v15;
      }
      else
      {
        v9 = 8 * v7;
        v10 = src;
        v11 = src;
        if ( *((_DWORD *)a1 + 98) )
        {
          memmove((void *)a1[48], src, 8 * v7);
          v11 = v15;
          v6 = (unsigned int)v16;
          v7 = v9;
          v10 = &v15[v9];
        }
      }
      v12 = 8 * v6;
      if ( v10 != &v11[v12] )
      {
        memcpy((void *)(v7 + a1[48]), v10, v12 - v7);
        v11 = v15;
      }
    }
    v13 = v18;
    *((_DWORD *)a1 + 98) = v8;
    *((_DWORD *)a1 + 112) = v13;
    if ( v11 != src )
      _libc_free((unsigned __int64)v11);
  }
  else
  {
    v14 = a1[48];
    if ( (_QWORD *)v14 != a1 + 50 )
    {
      _libc_free(v14);
      v5 = v15;
    }
    a1[48] = v5;
    a1[49] = v16;
    *((_DWORD *)a1 + 112) = v18;
  }
}
