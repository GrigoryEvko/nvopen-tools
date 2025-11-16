// Function: sub_2D3B550
// Address: 0x2d3b550
//
void __fastcall sub_2D3B550(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        int a4,
        __int64 a5,
        __int64 a6,
        int a7,
        __int64 *a8,
        _DWORD *a9)
{
  __int64 v10; // rdx
  int v12; // ebx
  unsigned int v13; // esi
  _DWORD *v14; // rax
  __int64 v15; // rsi
  int *v16; // rax
  int v17; // [rsp+10h] [rbp-B0h]
  int *v18; // [rsp+10h] [rbp-B0h]
  __int64 v20; // [rsp+28h] [rbp-98h] BYREF
  __int64 v21; // [rsp+30h] [rbp-90h] BYREF
  _BYTE *v22; // [rsp+38h] [rbp-88h]
  __int64 v23; // [rsp+40h] [rbp-80h]
  _BYTE v24[120]; // [rsp+48h] [rbp-78h] BYREF

  v17 = a6;
  if ( *(_BYTE *)(a1 + 24) )
  {
    v10 = (unsigned int)a9[48];
    v21 = (__int64)a9;
    v12 = a5;
    v22 = v24;
    v23 = 0x400000000LL;
    if ( (_DWORD)v10 )
    {
      sub_2D2BC70((__int64)&v21, a5, v10, (__int64)v24, a5, a6);
    }
    else
    {
      v13 = a9[49];
      if ( v13 )
      {
        v14 = a9 + 1;
        while ( (unsigned int)a5 >= *v14 )
        {
          v10 = (unsigned int)(v10 + 1);
          v14 += 2;
          if ( v13 == (_DWORD)v10 )
            goto LABEL_9;
        }
        v13 = v10;
      }
LABEL_9:
      sub_2D29C80((__int64)&v21, v13, v10, (__int64)v24, a5, a6);
    }
    if ( *(_DWORD *)sub_2D289F0((__int64)&v21) != v12 || *(_DWORD *)sub_2D28A10((__int64)&v21) != v17 )
    {
      v15 = *a8;
      v20 = v15;
      if ( v15 )
        sub_B96E90((__int64)&v20, v15, 1);
      v18 = (int *)sub_2D28A10((__int64)&v21);
      v16 = (int *)sub_2D289F0((__int64)&v21);
      sub_2D3AEE0(a1, a2, a3, a4, *v16, *v18, a7, &v20);
      if ( v20 )
        sub_B91220((__int64)&v20, v20);
    }
    if ( v22 != v24 )
      _libc_free((unsigned __int64)v22);
  }
}
