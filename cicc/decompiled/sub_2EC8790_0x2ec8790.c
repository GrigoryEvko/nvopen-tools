// Function: sub_2EC8790
// Address: 0x2ec8790
//
// bad sp value at call has been detected, the output may be wrong!
unsigned __int64 __fastcall sub_2EC8790(__int64 a1, int a2, char *a3, __int64 a4)
{
  __int64 v4; // r14
  char *v5; // r13
  __int64 v7; // rdi
  char v8; // al
  char v9; // al
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  const char *v14; // [rsp+0h] [rbp-60h] BYREF
  __int64 v15; // [rsp+8h] [rbp-58h]
  const char *v16; // [rsp+10h] [rbp-50h]
  __int16 v17; // [rsp+20h] [rbp-40h]

  v5 = a3;
  v7 = a1 + 24;
  v8 = a3[32];
  *(_QWORD *)(v7 - 24) = 0;
  *(_QWORD *)(v7 - 16) = 0;
  *(_QWORD *)(v7 - 8) = 0;
  if ( v8 )
  {
    if ( v8 == 1 )
    {
      v14 = ".A";
      v17 = 259;
    }
    else
    {
      if ( a3[33] == 1 )
      {
        a4 = *((_QWORD *)a3 + 1);
        a3 = *(char **)a3;
      }
      else
      {
        v8 = 2;
      }
      v15 = a4;
      v14 = a3;
      v16 = ".A";
      LOBYTE(v17) = v8;
      HIBYTE(v17) = 3;
    }
  }
  else
  {
    v17 = 256;
  }
  *(_DWORD *)(a1 + 24) = a2;
  sub_CA0F50((__int64 *)(v7 + 8), (void **)&v14);
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  v9 = v5[32];
  if ( v9 )
  {
    if ( v9 == 1 )
    {
      v14 = ".P";
      v17 = 259;
    }
    else
    {
      if ( v5[33] == 1 )
      {
        v4 = *((_QWORD *)v5 + 1);
        v5 = *(char **)v5;
      }
      else
      {
        v9 = 2;
      }
      v14 = v5;
      v15 = v4;
      v16 = ".P";
      LOBYTE(v17) = v9;
      HIBYTE(v17) = 3;
    }
  }
  else
  {
    v17 = 256;
  }
  *(_DWORD *)(a1 + 88) = 4 * a2;
  sub_CA0F50((__int64 *)(a1 + 96), (void **)&v14);
  *(_QWORD *)(a1 + 312) = a1 + 296;
  *(_QWORD *)(a1 + 320) = a1 + 296;
  *(_QWORD *)(a1 + 192) = a1 + 208;
  *(_QWORD *)(a1 + 360) = a1 + 376;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 200) = 0x1000000000LL;
  *(_DWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 304) = 0;
  *(_QWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 336) = 0;
  *(_QWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 368) = 0x1000000000LL;
  *(_QWORD *)(a1 + 440) = a1 + 456;
  *(_QWORD *)(a1 + 448) = 0x1000000000LL;
  return sub_2EC8570(a1, (__int64)&v14, a1 + 456, v10, v11, v12);
}
