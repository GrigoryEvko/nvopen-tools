// Function: sub_2DDBC50
// Address: 0x2ddbc50
//
void __fastcall sub_2DDBC50(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  _DWORD *v10; // rcx
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // r14
  __int64 v14; // rcx
  __int64 v15; // rdx
  unsigned __int64 v16; // rbx
  __int64 v17; // rdi
  __int64 v18; // [rsp+8h] [rbp-78h]
  __int64 v19; // [rsp+8h] [rbp-78h]
  unsigned __int64 v21[2]; // [rsp+20h] [rbp-60h] BYREF
  _BYTE v22[80]; // [rsp+30h] [rbp-50h] BYREF

  if ( a1 != a2 )
  {
    v6 = (__int64)(a1 + 6);
    if ( a2 != a1 + 6 )
    {
      while ( 1 )
      {
        v8 = *a1;
        v10 = *(_DWORD **)v6;
        v11 = *(unsigned int *)*a1;
        if ( **(_DWORD **)v6 < (unsigned int)v11
          || **(_DWORD **)v6 == (_DWORD)v11 && (v8 = *(unsigned int *)(v8 + 4), v10[1] < (unsigned int)v8) )
        {
          v12 = *(unsigned int *)(v6 + 8);
          v21[0] = (unsigned __int64)v22;
          v21[1] = 0x400000000LL;
          if ( (_DWORD)v12 )
          {
            v19 = v6;
            sub_2DDB710((__int64)v21, v6, v12, 0x400000000LL, a5, a6);
            v6 = v19;
          }
          v9 = v6 + 48;
          v13 = v6;
          v14 = 0xAAAAAAAAAAAAAAABLL;
          v15 = v6 - (_QWORD)a1;
          v16 = 0xAAAAAAAAAAAAAAABLL * ((v6 - (__int64)a1) >> 4);
          if ( v6 - (__int64)a1 > 0 )
          {
            do
            {
              v17 = v13;
              v13 -= 48;
              sub_2DDB710(v17, v13, v15, v14, a5, a6);
              --v16;
            }
            while ( v16 );
          }
          sub_2DDB710((__int64)a1, (__int64)v21, v15, v14, a5, a6);
          if ( (_BYTE *)v21[0] == v22 )
            goto LABEL_7;
          _libc_free(v21[0]);
          v6 = v9;
          if ( a2 == (__int64 *)v9 )
            return;
        }
        else
        {
          v18 = v6;
          sub_2DDBBA0(v6, v11, v8, (__int64)v10, a5, a6);
          v9 = v18 + 48;
LABEL_7:
          v6 = v9;
          if ( a2 == (__int64 *)v9 )
            return;
        }
      }
    }
  }
}
