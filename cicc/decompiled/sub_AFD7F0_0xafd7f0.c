// Function: sub_AFD7F0
// Address: 0xafd7f0
//
__int64 __fastcall sub_AFD7F0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // ebx
  __int64 v7; // r14
  int v8; // ebx
  int v9; // eax
  __int64 v10; // rsi
  _QWORD *v11; // rdi
  unsigned int v12; // eax
  int v13; // r8d
  _QWORD *v14; // rcx
  __int64 v15; // rdx
  _BYTE v16[8]; // [rsp+0h] [rbp-D0h] BYREF
  __int64 v17; // [rsp+8h] [rbp-C8h] BYREF
  __int64 v18; // [rsp+10h] [rbp-C0h] BYREF
  int v19; // [rsp+18h] [rbp-B8h] BYREF
  __int64 v20; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v21[4]; // [rsp+28h] [rbp-A8h] BYREF
  __int64 v22[3]; // [rsp+48h] [rbp-88h] BYREF
  __int64 v23[7]; // [rsp+60h] [rbp-70h] BYREF
  __int64 v24[7]; // [rsp+98h] [rbp-38h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v7 = *(_QWORD *)(a1 + 8);
    v8 = v4 - 1;
    sub_AF54F0((__int64)v16, *a2);
    v9 = sub_AFADE0(&v17, &v18, &v19, v21, &v20, v22, v23, v24);
    v10 = *a2;
    v11 = 0;
    v12 = v8 & v9;
    v13 = 1;
    v14 = (_QWORD *)(v7 + 8LL * v12);
    v15 = *v14;
    if ( *a2 == *v14 )
    {
LABEL_10:
      *a3 = v14;
      return 1;
    }
    else
    {
      while ( v15 != -4096 )
      {
        if ( v15 != -8192 || v11 )
          v14 = v11;
        v12 = v8 & (v13 + v12);
        v15 = *(_QWORD *)(v7 + 8LL * v12);
        if ( v15 == v10 )
        {
          v14 = (_QWORD *)(v7 + 8LL * v12);
          goto LABEL_10;
        }
        ++v13;
        v11 = v14;
        v14 = (_QWORD *)(v7 + 8LL * v12);
      }
      if ( !v11 )
        v11 = v14;
      *a3 = v11;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
