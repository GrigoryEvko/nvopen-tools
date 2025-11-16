// Function: sub_B541D0
// Address: 0xb541d0
//
__int64 __fastcall sub_B541D0(__int64 a1, void **a2, __int64 a3, __int64 a4)
{
  char v4; // r13
  int v5; // r12d
  __int64 result; // rax
  unsigned __int64 v8; // rcx
  __int64 v9; // rax
  int v10; // r13d
  unsigned int v11; // r13d
  __int64 v12; // rdi
  bool v13; // zf
  void *v14; // [rsp+10h] [rbp-60h] BYREF
  __int64 v15; // [rsp+18h] [rbp-58h]
  _BYTE s[80]; // [rsp+20h] [rbp-50h] BYREF

  v4 = BYTE4(a4);
  v5 = a4;
  result = sub_B53E30(*(_QWORD *)a1, (__int64)a2, a3);
  if ( !*(_BYTE *)(a1 + 56) )
  {
    if ( !v5 || !v4 )
      return result;
    v9 = *(_QWORD *)a1;
    *(_BYTE *)(a1 + 64) = 1;
    v10 = *(_DWORD *)(v9 + 4);
    v14 = s;
    v15 = 0x800000000LL;
    v11 = (v10 & 0x7FFFFFFu) >> 1;
    if ( v11 > 8 )
    {
      sub_C8D5F0(&v14, s, v11, 4);
      a2 = 0;
      memset(v14, 0, 4LL * v11);
      v13 = *(_BYTE *)(a1 + 56) == 0;
      LODWORD(v15) = v11;
      v12 = a1 + 8;
      if ( !v13 )
      {
        a2 = &v14;
        sub_B48480(v12, (char **)&v14);
LABEL_18:
        if ( v14 != s )
          _libc_free(v14, a2);
        result = *(_QWORD *)(a1 + 8);
        *(_DWORD *)(result + 4LL * (((*(_DWORD *)(*(_QWORD *)a1 + 4LL) & 0x7FFFFFFu) >> 1) - 1)) = v5;
        return result;
      }
    }
    else
    {
      if ( v11 && 4LL * v11 )
      {
        a2 = 0;
        memset(s, 0, 4LL * v11);
      }
      LODWORD(v15) = v11;
      v12 = a1 + 8;
    }
    *(_QWORD *)(a1 + 8) = a1 + 24;
    *(_QWORD *)(a1 + 16) = 0x800000000LL;
    if ( (_DWORD)v15 )
    {
      a2 = &v14;
      sub_B48480(v12, (char **)&v14);
    }
    *(_BYTE *)(a1 + 56) = 1;
    goto LABEL_18;
  }
  v8 = *(unsigned int *)(a1 + 20);
  *(_BYTE *)(a1 + 64) = 1;
  if ( !v4 )
    v5 = 0;
  result = *(unsigned int *)(a1 + 16);
  if ( result + 1 > v8 )
  {
    sub_C8D5F0(a1 + 8, a1 + 24, result + 1, 4);
    result = *(unsigned int *)(a1 + 16);
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 8) + 4 * result) = v5;
  ++*(_DWORD *)(a1 + 16);
  return result;
}
