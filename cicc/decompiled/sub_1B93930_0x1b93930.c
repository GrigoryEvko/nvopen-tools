// Function: sub_1B93930
// Address: 0x1b93930
//
__int64 *__fastcall sub_1B93930(__int64 *a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  _QWORD *v6; // rbx
  _QWORD *v7; // r13
  __int64 v8; // r15
  __int64 v9; // rdi
  _QWORD v11[4]; // [rsp+0h] [rbp-A0h] BYREF
  char *v12; // [rsp+20h] [rbp-80h]
  __int64 v13; // [rsp+28h] [rbp-78h]
  char v14; // [rsp+30h] [rbp-70h] BYREF
  __int64 v15; // [rsp+38h] [rbp-68h]
  _QWORD *v16; // [rsp+40h] [rbp-60h]
  __int64 v17; // [rsp+48h] [rbp-58h]
  unsigned int v18; // [rsp+50h] [rbp-50h]
  __int64 v19; // [rsp+60h] [rbp-40h]
  char v20; // [rsp+68h] [rbp-38h]
  int v21; // [rsp+6Ch] [rbp-34h]

  v3 = sub_22077B0(472);
  if ( v3 )
  {
    *(_QWORD *)v3 = 0;
    *(_QWORD *)(v3 + 8) = v3 + 24;
    *(_QWORD *)(v3 + 56) = v3 + 40;
    *(_QWORD *)(v3 + 64) = v3 + 40;
    *(_QWORD *)(v3 + 80) = v3 + 96;
    *(_QWORD *)(v3 + 120) = v3 + 152;
    *(_QWORD *)(v3 + 128) = v3 + 152;
    *(_QWORD *)(v3 + 16) = 0x200000000LL;
    *(_QWORD *)(v3 + 384) = v3 + 400;
    *(_DWORD *)(v3 + 40) = 0;
    *(_QWORD *)(v3 + 48) = 0;
    *(_QWORD *)(v3 + 72) = 0;
    *(_QWORD *)(v3 + 88) = 0;
    *(_BYTE *)(v3 + 96) = 0;
    *(_QWORD *)(v3 + 112) = 0;
    *(_QWORD *)(v3 + 136) = 16;
    *(_DWORD *)(v3 + 144) = 0;
    *(_QWORD *)(v3 + 280) = 0;
    *(_QWORD *)(v3 + 288) = 0;
    *(_QWORD *)(v3 + 296) = 0;
    *(_DWORD *)(v3 + 304) = 0;
    *(_QWORD *)(v3 + 312) = 0;
    *(_QWORD *)(v3 + 320) = 0;
    *(_QWORD *)(v3 + 328) = 0;
    *(_DWORD *)(v3 + 336) = 0;
    *(_QWORD *)(v3 + 344) = 0;
    *(_QWORD *)(v3 + 352) = 0;
    *(_QWORD *)(v3 + 360) = 0;
    *(_QWORD *)(v3 + 368) = 0;
    *(_QWORD *)(v3 + 376) = 0;
    *(_QWORD *)(v3 + 392) = 0x400000000LL;
    *(_QWORD *)(v3 + 432) = v3 + 448;
    *(_QWORD *)(v3 + 440) = 0;
    *(_QWORD *)(v3 + 448) = 0;
    *(_QWORD *)(v3 + 456) = 1;
  }
  *a1 = v3;
  v4 = a2[1];
  v5 = *a2;
  v11[2] = v3;
  v11[1] = v4;
  v11[0] = v5;
  v12 = &v14;
  v13 = 0x100000000LL;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  ((void (__fastcall *)(_QWORD *))sub_1BEFB10)(v11);
  if ( v18 )
  {
    v6 = v16;
    v7 = &v16[2 * v18];
    do
    {
      if ( *v6 != -16 && *v6 != -8 )
      {
        v8 = v6[1];
        if ( v8 )
        {
          v9 = *(_QWORD *)(v8 + 24);
          if ( v9 )
            j_j___libc_free_0(v9, *(_QWORD *)(v8 + 40) - v9);
          j_j___libc_free_0(v8, 56);
        }
      }
      v6 += 2;
    }
    while ( v7 != v6 );
  }
  j___libc_free_0(v16);
  if ( v12 != &v14 )
    _libc_free((unsigned __int64)v12);
  return a1;
}
