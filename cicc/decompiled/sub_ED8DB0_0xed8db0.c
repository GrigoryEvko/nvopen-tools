// Function: sub_ED8DB0
// Address: 0xed8db0
//
_QWORD *__fastcall sub_ED8DB0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r15
  __int64 v7; // r14
  _QWORD *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r9
  _QWORD *v12; // r8
  unsigned int v13; // ebx
  void *v14; // rdi
  unsigned int v15; // ebx
  void *v16; // rdi
  size_t v18; // rdx
  size_t v19; // rdx
  _QWORD *v20; // [rsp+0h] [rbp-40h]
  _QWORD *v21; // [rsp+0h] [rbp-40h]
  _QWORD *v22; // [rsp+0h] [rbp-40h]
  _QWORD *v23; // [rsp+0h] [rbp-40h]
  _QWORD *v24; // [rsp+0h] [rbp-40h]

  v6 = *a1;
  v7 = a1[1];
  v8 = (_QWORD *)sub_22077B0(536);
  v12 = v8;
  if ( v8 )
  {
    *v8 = v6;
    v8[1] = v7;
    v8[2] = a1 + 2;
    v8[3] = a3;
    v8[4] = *(_QWORD *)a4;
    v8[5] = v8 + 7;
    v8[6] = 0x1C00000000LL;
    if ( *(_DWORD *)(a4 + 16) )
    {
      v20 = v8;
      sub_ED6600((__int64)(v8 + 5), a4 + 8, v9, v10, (__int64)v8, v11);
      v12 = v20;
    }
    v13 = *(_DWORD *)(a4 + 256);
    v14 = v12 + 37;
    v12[35] = v12 + 37;
    v12[36] = 0x100000000LL;
    if ( v13 && v12 + 35 != (_QWORD *)(a4 + 248) )
    {
      v19 = 168;
      if ( v13 == 1
        || (v23 = v12,
            sub_C8D5F0((__int64)(v12 + 35), v12 + 37, v13, 0xA8u, (__int64)v12, v11),
            v12 = v23,
            v14 = (void *)v23[35],
            (v19 = 168LL * *(unsigned int *)(a4 + 256)) != 0) )
      {
        v22 = v12;
        memcpy(v14, *(const void **)(a4 + 248), v19);
        v12 = v22;
      }
      *((_DWORD *)v12 + 72) = v13;
    }
    v15 = *(_DWORD *)(a4 + 440);
    v16 = v12 + 60;
    v12[58] = v12 + 60;
    v12[59] = 0x600000000LL;
    if ( v15 && v12 + 58 != (_QWORD *)(a4 + 432) )
    {
      v18 = 8LL * v15;
      if ( v15 <= 6
        || (v24 = v12,
            sub_C8D5F0((__int64)(v12 + 58), v12 + 60, v15, 8u, (__int64)v12, v11),
            v12 = v24,
            v18 = 8LL * *(unsigned int *)(a4 + 440),
            v16 = (void *)v24[58],
            v18) )
      {
        v21 = v12;
        memcpy(v16, *(const void **)(a4 + 432), v18);
        v12 = v21;
      }
      *((_DWORD *)v12 + 118) = v15;
    }
    v12[66] = a2;
  }
  return v12;
}
