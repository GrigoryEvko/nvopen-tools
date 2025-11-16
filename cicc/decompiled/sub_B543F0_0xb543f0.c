// Function: sub_B543F0
// Address: 0xb543f0
//
__int64 __fastcall sub_B543F0(__int64 a1, void **a2, unsigned __int64 a3)
{
  __int64 result; // rax
  unsigned int v5; // r12d
  int v6; // r14d
  unsigned int v7; // r14d
  size_t v8; // rdx
  __int64 v9; // rdi
  bool v10; // zf
  int v11; // [rsp+18h] [rbp-68h]
  void *v12; // [rsp+20h] [rbp-60h] BYREF
  __int64 v13; // [rsp+28h] [rbp-58h]
  _BYTE s[80]; // [rsp+30h] [rbp-50h] BYREF

  result = HIDWORD(a3);
  v11 = a3;
  if ( BYTE4(a3) )
  {
    result = *(unsigned __int8 *)(a1 + 56);
    v5 = (unsigned int)a2;
    if ( (_BYTE)result == 1 )
    {
LABEL_5:
      result = *(_QWORD *)(a1 + 8) + 4LL * v5;
      if ( v11 != *(_DWORD *)result )
      {
        *(_BYTE *)(a1 + 64) = 1;
        *(_DWORD *)result = v11;
      }
      return result;
    }
    if ( !(_DWORD)a3 )
    {
LABEL_4:
      if ( !(_BYTE)result )
        return result;
      goto LABEL_5;
    }
    v6 = *(_DWORD *)(*(_QWORD *)a1 + 4LL);
    v12 = s;
    v13 = 0x800000000LL;
    v7 = (v6 & 0x7FFFFFFu) >> 1;
    if ( v7 > 8 )
    {
      sub_C8D5F0(&v12, s, v7, 4);
      a2 = 0;
      memset(v12, 0, 4LL * v7);
      v10 = *(_BYTE *)(a1 + 56) == 0;
      LODWORD(v13) = v7;
      v9 = a1 + 8;
      if ( !v10 )
      {
        a2 = &v12;
        sub_B48480(v9, (char **)&v12);
LABEL_16:
        if ( v12 != s )
          _libc_free(v12, a2);
        result = *(unsigned __int8 *)(a1 + 56);
        goto LABEL_4;
      }
    }
    else
    {
      if ( v7 )
      {
        v8 = 4LL * v7;
        if ( v8 )
        {
          a2 = 0;
          memset(s, 0, v8);
        }
      }
      LODWORD(v13) = v7;
      v9 = a1 + 8;
    }
    *(_QWORD *)(a1 + 8) = a1 + 24;
    *(_QWORD *)(a1 + 16) = 0x800000000LL;
    if ( (_DWORD)v13 )
    {
      a2 = &v12;
      sub_B48480(v9, (char **)&v12);
    }
    *(_BYTE *)(a1 + 56) = 1;
    goto LABEL_16;
  }
  return result;
}
