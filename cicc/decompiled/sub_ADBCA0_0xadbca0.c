// Function: sub_ADBCA0
// Address: 0xadbca0
//
__int64 __fastcall sub_ADBCA0(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // eax
  __int64 v5; // rax
  __int64 v6; // r9
  __int64 v7; // r8
  unsigned __int64 v8; // rcx
  __int64 v9; // rdx
  unsigned int v10; // r14d
  int v11; // r15d
  __int64 v12; // r11
  __int64 v13; // rax
  _BYTE *v14; // rsi
  __int64 result; // rax
  __int64 v16; // rdx
  _QWORD *v17; // rax
  __int64 v18; // [rsp+0h] [rbp-A0h]
  __int64 v19; // [rsp+8h] [rbp-98h]
  __int64 v20; // [rsp+10h] [rbp-90h]
  __int64 v22; // [rsp+18h] [rbp-88h]
  _BYTE *v23; // [rsp+20h] [rbp-80h] BYREF
  __int64 v24; // [rsp+28h] [rbp-78h]
  _BYTE v25[112]; // [rsp+30h] [rbp-70h] BYREF

  v24 = 0x800000000LL;
  v4 = *(_DWORD *)(a1 + 4);
  v23 = v25;
  v5 = v4 & 0x7FFFFFF;
  if ( (_DWORD)v5 )
  {
    v6 = (unsigned int)(v5 - 1);
    v7 = 0;
    v8 = 8;
    v9 = 0;
    v10 = 0;
    v11 = 0;
    while ( 1 )
    {
      v13 = *(_QWORD *)(a1 + 32 * (v7 - v5));
      if ( a2 == v13 )
      {
        v12 = v9 + 1;
        v13 = a3;
        ++v11;
        v10 = v7;
        if ( v9 + 1 <= v8 )
          goto LABEL_4;
      }
      else
      {
        v12 = v9 + 1;
        if ( v9 + 1 <= v8 )
          goto LABEL_4;
      }
      v18 = v7;
      v19 = v6;
      v20 = v13;
      sub_C8D5F0(&v23, v25, v12, 8);
      v9 = (unsigned int)v24;
      v7 = v18;
      v6 = v19;
      v13 = v20;
LABEL_4:
      *(_QWORD *)&v23[8 * v9] = v13;
      v9 = (unsigned int)(v24 + 1);
      LODWORD(v24) = v24 + 1;
      if ( v6 == v7 )
      {
        v14 = v23;
        goto LABEL_10;
      }
      v8 = HIDWORD(v24);
      ++v7;
      v5 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
    }
  }
  v11 = 0;
  v10 = 0;
  v9 = 0;
  v14 = v25;
LABEL_10:
  result = sub_ADABF0(a1, (__int64)v14, v9, *(__int64 ***)(a1 + 8), 1, 0);
  if ( !result )
  {
    v17 = (_QWORD *)sub_BD5C60(a1, v14, v16);
    v14 = v23;
    result = sub_ADB270(*v17 + 2120LL, (__int64)v23, (unsigned int)v24, a1, a2, a3, v11, v10);
  }
  if ( v23 != v25 )
  {
    v22 = result;
    _libc_free(v23, v14);
    return v22;
  }
  return result;
}
