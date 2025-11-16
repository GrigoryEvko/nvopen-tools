// Function: sub_CFC580
// Address: 0xcfc580
//
__int64 __fastcall sub_CFC580(__int64 *a1, unsigned __int8 *a2, int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v8; // r12
  __int64 v9; // rbx
  unsigned __int64 v10; // rsi
  int v11; // edx
  __int64 v12; // rax
  char *v13; // r14
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // r15
  _QWORD v17[2]; // [rsp+0h] [rbp-50h] BYREF
  unsigned __int8 *v18; // [rsp+10h] [rbp-40h]
  int v19; // [rsp+18h] [rbp-38h]

  result = *a2;
  if ( (unsigned __int8)(result - 4) > 0x18u || (_BYTE)result == 22 )
  {
    v17[0] = 4;
    v8 = *a1;
    v17[1] = 0;
    v18 = a2;
    if ( a2 != (unsigned __int8 *)-4096LL && a2 != (unsigned __int8 *)-8192LL )
      sub_BD73F0((__int64)v17);
    v19 = a3;
    v9 = *(unsigned int *)(v8 + 8);
    v10 = v9 + 1;
    v11 = v9;
    if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(v8 + 12) )
    {
      v16 = *(_QWORD *)v8;
      if ( *(_QWORD *)v8 > (unsigned __int64)v17 || (unsigned __int64)v17 >= v16 + 32 * v9 )
      {
        sub_CFC2E0(v8, v10, v9, a4, a5, a6);
        v9 = *(unsigned int *)(v8 + 8);
        v12 = *(_QWORD *)v8;
        v13 = (char *)v17;
        v11 = *(_DWORD *)(v8 + 8);
      }
      else
      {
        sub_CFC2E0(v8, v10, v9, a4, a5, a6);
        v12 = *(_QWORD *)v8;
        v9 = *(unsigned int *)(v8 + 8);
        v13 = (char *)v17 + *(_QWORD *)v8 - v16;
        v11 = *(_DWORD *)(v8 + 8);
      }
    }
    else
    {
      v12 = *(_QWORD *)v8;
      v13 = (char *)v17;
    }
    v14 = v12 + 32 * v9;
    if ( v14 )
    {
      *(_QWORD *)v14 = 4;
      v15 = *((_QWORD *)v13 + 2);
      *(_QWORD *)(v14 + 8) = 0;
      *(_QWORD *)(v14 + 16) = v15;
      if ( v15 != -4096 && v15 != 0 && v15 != -8192 )
        sub_BD6050((unsigned __int64 *)v14, *(_QWORD *)v13 & 0xFFFFFFFFFFFFFFF8LL);
      *(_DWORD *)(v14 + 24) = *((_DWORD *)v13 + 6);
      v11 = *(_DWORD *)(v8 + 8);
    }
    *(_DWORD *)(v8 + 8) = v11 + 1;
    result = (__int64)v18;
    if ( v18 != 0 && v18 + 4096 != 0 && v18 != (unsigned __int8 *)-8192LL )
      return sub_BD60C0(v17);
  }
  return result;
}
