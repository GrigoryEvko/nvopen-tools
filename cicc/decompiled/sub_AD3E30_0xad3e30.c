// Function: sub_AD3E30
// Address: 0xad3e30
//
__int64 __fastcall sub_AD3E30(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rsi
  int v5; // edx
  __int64 v6; // rax
  __int64 v7; // rdx
  int v8; // r9d
  __int64 v9; // r11
  __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // r15
  __int64 v13; // rdx
  __int64 v14; // r15
  _QWORD *v16; // rax
  __int64 v17; // [rsp+8h] [rbp-A8h]
  int v18; // [rsp+14h] [rbp-9Ch]
  int v19; // [rsp+18h] [rbp-98h]
  unsigned int v20; // [rsp+20h] [rbp-90h]
  __int64 *v22; // [rsp+30h] [rbp-80h] BYREF
  __int64 v23; // [rsp+38h] [rbp-78h]
  _BYTE v24[112]; // [rsp+40h] [rbp-70h] BYREF

  v4 = 0;
  v5 = *(_DWORD *)(a1 + 4);
  v23 = 0x800000000LL;
  v6 = 0;
  v7 = v5 & 0x7FFFFFF;
  v22 = (__int64 *)v24;
  if ( (unsigned int)v7 > 8uLL )
  {
    sub_C8D5F0(&v22, v24, (unsigned int)v7, 8);
    v4 = (unsigned int)v23;
    v7 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
    v6 = (unsigned int)v23;
    v8 = v7;
    if ( (_DWORD)v7 )
      goto LABEL_3;
  }
  else
  {
    v8 = v7;
    if ( (_DWORD)v7 )
    {
LABEL_3:
      v20 = 0;
      v9 = (unsigned int)(v7 - 1);
      v10 = 0;
      v8 = 0;
      while ( 1 )
      {
        v12 = *(_QWORD *)(a1 + 32 * (v10 - v7));
        if ( a2 == v12 )
        {
          v11 = v6 + 1;
          ++v8;
          v20 = v10;
          v12 = a3;
          if ( v6 + 1 <= (unsigned __int64)HIDWORD(v23) )
            goto LABEL_5;
        }
        else
        {
          v11 = v6 + 1;
          if ( v6 + 1 <= (unsigned __int64)HIDWORD(v23) )
            goto LABEL_5;
        }
        v17 = v9;
        v18 = v8;
        sub_C8D5F0(&v22, v24, v11, 8);
        v6 = (unsigned int)v23;
        v9 = v17;
        v8 = v18;
LABEL_5:
        v22[v6] = v12;
        v6 = (unsigned int)(v23 + 1);
        LODWORD(v23) = v23 + 1;
        if ( v9 == v10 )
        {
          v4 = (unsigned int)v6;
          goto LABEL_11;
        }
        ++v10;
        v7 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
      }
    }
  }
  v20 = 0;
LABEL_11:
  v19 = v8;
  v14 = sub_ACE990(v22, v4);
  if ( !v14 )
  {
    v16 = (_QWORD *)sub_BD5C60(a1, v4, v13);
    v4 = (__int64)v22;
    v14 = sub_AD3780(*v16 + 1808LL, v22, (unsigned int)v23, a1, a2, a3, v19, v20);
  }
  if ( v22 != (__int64 *)v24 )
    _libc_free(v22, v4);
  return v14;
}
