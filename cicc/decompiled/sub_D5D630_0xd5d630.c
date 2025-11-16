// Function: sub_D5D630
// Address: 0xd5d630
//
__int64 __fastcall sub_D5D630(__int64 a1, __int64 a2, const void **a3, unsigned __int16 a4)
{
  unsigned int v6; // edx
  unsigned __int64 v7; // rsi
  __int64 v8; // rax
  _QWORD *v10; // rdi
  __int64 v11; // [rsp+0h] [rbp-20h] BYREF
  unsigned int v12; // [rsp+8h] [rbp-18h]

  v6 = *((_DWORD *)a3 + 2);
  if ( *(_BYTE *)(a2 + 17) && HIBYTE(a4) )
  {
    v10 = *a3;
    if ( v6 > 0x40 )
      v10 = (_QWORD *)*v10;
    v12 = *(_DWORD *)(a2 + 32);
    if ( v12 > 0x40 )
    {
      sub_C43690((__int64)&v11, ((unsigned __int64)v10 + (1LL << a4) - 1) & -(1LL << a4), 0);
      v6 = *((_DWORD *)a3 + 2);
    }
    else
    {
      v11 = ((unsigned __int64)v10 + (1LL << a4) - 1) & -(1LL << a4);
    }
    if ( v6 > 0x40 )
    {
      if ( *a3 )
        j_j___libc_free_0_0(*a3);
    }
    v6 = v12;
    *a3 = (const void *)v11;
    *((_DWORD *)a3 + 2) = v6;
  }
  v7 = (unsigned __int64)*a3;
  v8 = 1LL << ((unsigned __int8)v6 - 1);
  if ( v6 > 0x40 )
  {
    if ( (*(_QWORD *)(v7 + 8LL * ((v6 - 1) >> 6)) & v8) != 0 )
      goto LABEL_5;
    *(_DWORD *)(a1 + 8) = v6;
    sub_C43780(a1, a3);
    return a1;
  }
  else
  {
    if ( (v8 & v7) != 0 )
    {
LABEL_5:
      *(_DWORD *)(a1 + 8) = 1;
      *(_QWORD *)a1 = 0;
      return a1;
    }
    *(_DWORD *)(a1 + 8) = v6;
    *(_QWORD *)a1 = v7;
    return a1;
  }
}
