// Function: sub_1D23470
// Address: 0x1d23470
//
__int64 __fastcall sub_1D23470(__int64 a1, __int64 a2, __int64 a3, int a4, int a5, int a6)
{
  __int64 v6; // r12
  __int64 v7; // rdx
  unsigned __int64 v8; // rdi
  _QWORD *v9; // rax
  unsigned __int64 v11[2]; // [rsp+0h] [rbp-30h] BYREF
  int v12; // [rsp+10h] [rbp-20h]

  v6 = a1;
  v7 = *(unsigned __int16 *)(a1 + 24);
  if ( (_WORD)v7 != 11 && (_DWORD)v7 != 33 )
  {
    v6 = 0;
    if ( (_WORD)v7 == 104 )
    {
      v11[0] = 0;
      v11[1] = 0;
      v12 = 0;
      v6 = sub_1D23440(a1, (__int64)v11, v7, a4, a5, a6);
      if ( v6 )
      {
        v8 = v11[0];
        if ( !((unsigned int)(v12 + 63) >> 6) )
        {
LABEL_11:
          _libc_free(v11[0]);
          return v6;
        }
        v9 = (_QWORD *)v11[0];
        while ( !*v9 )
        {
          if ( (_QWORD *)(v11[0] + 8LL * (((unsigned int)(v12 + 63) >> 6) - 1) + 8) == ++v9 )
            goto LABEL_11;
        }
      }
      else
      {
        v8 = v11[0];
      }
      _libc_free(v8);
      return 0;
    }
  }
  return v6;
}
