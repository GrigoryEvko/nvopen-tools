// Function: sub_3103310
// Address: 0x3103310
//
__int64 __fastcall sub_3103310(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r12
  __int64 v4; // rsi
  _QWORD *v5; // r12
  _QWORD *v6; // r13
  __int64 v7; // rax
  unsigned __int64 *v8; // rax
  unsigned __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // [rsp+0h] [rbp-40h] BYREF
  __int64 v12; // [rsp+8h] [rbp-38h]
  __int64 v13; // [rsp+10h] [rbp-30h]
  int v14; // [rsp+18h] [rbp-28h]

  result = **(_QWORD **)(a2 + 32);
  v3 = *(_QWORD *)(result + 72);
  if ( (*(_BYTE *)(v3 + 2) & 8) != 0 )
  {
    result = sub_B2E500(*(_QWORD *)(result + 72));
    if ( result )
    {
      result = sub_B2A630(result);
      if ( (int)result > 10 )
      {
        if ( (_DWORD)result != 12 )
          return result;
      }
      else if ( (int)result <= 6 )
      {
        return result;
      }
      sub_B2AF20((__int64)&v11, v3);
      v4 = *(unsigned int *)(a1 + 32);
      if ( (_DWORD)v4 )
      {
        v5 = *(_QWORD **)(a1 + 16);
        v6 = &v5[2 * v4];
        do
        {
          if ( *v5 != -8192 && *v5 != -4096 )
          {
            v7 = v5[1];
            if ( v7 )
            {
              if ( (v7 & 4) != 0 )
              {
                v8 = (unsigned __int64 *)(v7 & 0xFFFFFFFFFFFFFFF8LL);
                v9 = (unsigned __int64)v8;
                if ( v8 )
                {
                  if ( (unsigned __int64 *)*v8 != v8 + 2 )
                    _libc_free(*v8);
                  j_j___libc_free_0(v9);
                }
              }
            }
          }
          v5 += 2;
        }
        while ( v6 != v5 );
        v4 = *(unsigned int *)(a1 + 32);
      }
      sub_C7D6A0(*(_QWORD *)(a1 + 16), 16 * v4, 8);
      v10 = v12;
      ++*(_QWORD *)(a1 + 8);
      ++v11;
      *(_QWORD *)(a1 + 16) = v10;
      v12 = 0;
      *(_QWORD *)(a1 + 24) = v13;
      v13 = 0;
      *(_DWORD *)(a1 + 32) = v14;
      v14 = 0;
      return sub_C7D6A0(0, 0, 8);
    }
  }
  return result;
}
