// Function: sub_28CC2D0
// Address: 0x28cc2d0
//
__int64 __fastcall sub_28CC2D0(__int64 a1, _BYTE *a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  unsigned int v7; // edx
  _QWORD *v8; // rbx
  _BYTE *v9; // rsi
  __int64 v10; // r15
  _QWORD *v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rdx
  int v15; // r8d
  _QWORD v16[11]; // [rsp-58h] [rbp-58h] BYREF

  v3 = *(unsigned int *)(a1 + 1944);
  v4 = *(_QWORD *)(a1 + 1928);
  if ( !(_DWORD)v3 )
    return 0;
  v7 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (_QWORD *)(v4 + 16LL * v7);
  v9 = (_BYTE *)*v8;
  if ( a2 != (_BYTE *)*v8 )
  {
    v15 = 1;
    while ( v9 != (_BYTE *)-4096LL )
    {
      v7 = (v3 - 1) & (v15 + v7);
      v8 = (_QWORD *)(v4 + 16LL * v7);
      v9 = (_BYTE *)*v8;
      if ( a2 == (_BYTE *)*v8 )
        goto LABEL_3;
      ++v15;
    }
    return 0;
  }
LABEL_3:
  if ( v8 == (_QWORD *)(v4 + 16 * v3) )
    return 0;
  v10 = v8[1];
  if ( v10 == a3 )
    return 0;
  if ( *a2 == 28 )
  {
    sub_25DDDB0(v10 + 128, (__int64)a2);
    v11 = sub_AE6EC0(a3 + 128, (__int64)a2);
    v12 = *(_BYTE *)(a3 + 156) ? *(unsigned int *)(a3 + 148) : *(unsigned int *)(a3 + 144);
    v13 = *(_QWORD *)(a3 + 136) + 8 * v12;
    v16[0] = v11;
    v16[1] = v13;
    sub_254BBF0((__int64)v16);
    if ( a2 == *(_BYTE **)(v10 + 48) )
    {
      if ( *(_DWORD *)(v10 + 176) || *(_DWORD *)(v10 + 148) != *(_DWORD *)(v10 + 152) )
      {
        *(_QWORD *)(v10 + 48) = sub_28CBF00(a1, v10);
        sub_28CABC0(a1, v10);
      }
      else
      {
        *(_QWORD *)(v10 + 48) = 0;
      }
    }
  }
  v8[1] = a3;
  return 1;
}
