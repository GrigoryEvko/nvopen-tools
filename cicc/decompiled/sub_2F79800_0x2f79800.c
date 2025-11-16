// Function: sub_2F79800
// Address: 0x2f79800
//
__int64 __fastcall sub_2F79800(__int64 a1, __int64 a2)
{
  __int64 *v3; // r13
  __int64 v5; // rdi
  unsigned int v6; // eax
  _DWORD *v7; // rcx
  __int64 v8; // r8
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 result; // rax
  __int64 i; // r14
  unsigned int v13; // r10d
  __int64 v14; // rdx
  __int64 v15; // rsi
  unsigned int v16; // eax
  char v17; // r9
  __int64 v18; // rdi
  __int64 (*v19)(); // rax
  char v20; // al
  __int64 v21; // [rsp-10h] [rbp-60h]
  __int64 v22; // [rsp-10h] [rbp-60h]
  __int64 v23; // [rsp-8h] [rbp-58h]
  unsigned int v24; // [rsp+Ch] [rbp-44h]
  int v25[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v3 = (__int64 *)(a1 + 392);
  v5 = *(_QWORD *)(a1 + 8);
  v25[0] = 0;
  v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 392LL))(v5);
  sub_1D05C60((__int64)v3, v6, v25);
  v9 = *(_QWORD *)(a1 + 48);
  v10 = *(_QWORD *)(v9 + 232);
  result = 3LL * *(unsigned int *)(v9 + 240);
  for ( i = v10 + 8 * result; i != v10; result = v22 )
  {
    while ( 1 )
    {
      v13 = *(_DWORD *)v10;
      if ( *(int *)v10 < 0 )
      {
        v14 = *(_QWORD *)(a2 + 376);
        v15 = *(unsigned int *)(a2 + 336);
        v16 = *(unsigned __int8 *)(v14 + (v13 & 0x7FFFFFFF));
        if ( v16 >= (unsigned int)v15 )
          break;
        v8 = *(_QWORD *)(a2 + 328);
        while ( 1 )
        {
          v7 = (_DWORD *)(v8 + 4LL * v16);
          v14 = *v7 & 0x7FFFFFFF;
          if ( (v13 & 0x7FFFFFFF) == (_DWORD)v14 )
            break;
          v16 += 256;
          if ( (unsigned int)v15 <= v16 )
            goto LABEL_10;
        }
        result = v8 + 4 * v15;
        if ( v7 == (_DWORD *)result )
          break;
      }
      v10 += 24;
      if ( i == v10 )
        return result;
    }
LABEL_10:
    v17 = 0;
    if ( *(_BYTE *)(a1 + 58) )
    {
      v18 = *(_QWORD *)(a1 + 8);
      v17 = 0;
      v19 = *(__int64 (**)())(*(_QWORD *)v18 + 432LL);
      if ( v19 != sub_2F73F20 )
      {
        v24 = *(_DWORD *)v10;
        v20 = ((__int64 (__fastcall *)(__int64, __int64, __int64, _DWORD *, __int64, _QWORD))v19)(
                v18,
                v15,
                v14,
                v7,
                v8,
                0);
        v13 = v24;
        v17 = v20;
      }
    }
    v23 = *(_QWORD *)(v10 + 16);
    v21 = *(_QWORD *)(v10 + 8);
    v10 += 24;
    sub_2F74AE0(v3, *(_QWORD **)(a1 + 24), v13, 0, 0, v17, v21, v23);
  }
  return result;
}
