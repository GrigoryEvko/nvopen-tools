// Function: sub_ADC590
// Address: 0xadc590
//
__int64 __fastcall sub_ADC590(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rsi
  unsigned int v6; // edx
  __int64 *v7; // rcx
  __int64 v8; // rdi
  unsigned __int64 v9; // rdx
  _QWORD *v10; // rbx
  __int64 v11; // r12
  int v12; // r8d
  _QWORD *v13; // rax
  int v14; // edx
  _BYTE *v15; // rsi
  _QWORD *v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // rax
  int v19; // ecx
  int v20; // r9d
  int v21; // [rsp+8h] [rbp-C8h]
  _BYTE *v22; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v23; // [rsp+18h] [rbp-B8h]
  _BYTE v24[176]; // [rsp+20h] [rbp-B0h] BYREF

  result = *(unsigned int *)(a1 + 424);
  v4 = *(_QWORD *)(a1 + 408);
  if ( (_DWORD)result )
  {
    v6 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v4 + 56LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
    {
LABEL_3:
      result = v4 + 56 * result;
      if ( v7 != (__int64 *)result )
      {
        v9 = *((unsigned int *)v7 + 4);
        v10 = (_QWORD *)v7[1];
        v22 = v24;
        v23 = 0x1000000000LL;
        v11 = v9;
        v12 = v9;
        if ( v9 > 0x10 )
        {
          v21 = v9;
          sub_C8D5F0(&v22, v24, v9, 8);
          v15 = v22;
          v14 = v23;
          v12 = v21;
          v13 = &v22[8 * (unsigned int)v23];
        }
        else
        {
          v13 = v24;
          v14 = 0;
          v15 = v24;
        }
        if ( v11 * 8 )
        {
          v16 = &v13[v11];
          do
          {
            if ( v13 )
              *v13 = *v10;
            ++v13;
            ++v10;
          }
          while ( v13 != v16 );
          v15 = v22;
          v14 = v23;
        }
        v17 = *(_QWORD *)(a1 + 8);
        LODWORD(v23) = v12 + v14;
        v18 = sub_B9C770(v17, v15, (unsigned int)(v12 + v14), 0, 1);
        result = sub_BA6610(a2, 7, v18);
        if ( v22 != v24 )
          return _libc_free(v22, 7);
      }
    }
    else
    {
      v19 = 1;
      while ( v8 != -4096 )
      {
        v20 = v19 + 1;
        v6 = (result - 1) & (v19 + v6);
        v7 = (__int64 *)(v4 + 56LL * v6);
        v8 = *v7;
        if ( a2 == *v7 )
          goto LABEL_3;
        v19 = v20;
      }
    }
  }
  return result;
}
