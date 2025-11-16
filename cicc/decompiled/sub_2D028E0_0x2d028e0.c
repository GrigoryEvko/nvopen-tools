// Function: sub_2D028E0
// Address: 0x2d028e0
//
_QWORD *__fastcall sub_2D028E0(__int64 *a1)
{
  _QWORD *v1; // rax
  _QWORD *v2; // r12
  int v3; // eax
  __int128 *v4; // rax
  __int64 v6; // r10
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rax
  size_t v13; // rdx
  __int64 v14; // r9
  __int64 v15; // r8
  size_t *v16; // r9
  size_t **v17; // r14
  __int64 v18; // [rsp+0h] [rbp-60h]
  __int64 v19; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+10h] [rbp-50h]
  __int64 v21; // [rsp+18h] [rbp-48h]
  __int64 v22; // [rsp+20h] [rbp-40h]
  size_t n; // [rsp+28h] [rbp-38h]

  v1 = (_QWORD *)sub_22077B0(0xC8u);
  v2 = v1;
  if ( v1 )
  {
    v1[1] = 0;
    v1[2] = &unk_5014D64;
    v1[7] = v1 + 13;
    v1[14] = v1 + 20;
    *v1 = off_4A25BD8;
    v1[24] = 0x1000000000LL;
    v3 = *((_DWORD *)a1 + 3);
    *((_DWORD *)v2 + 6) = 4;
    v2[4] = 0;
    v2[5] = 0;
    v2[6] = 0;
    v2[8] = 1;
    v2[9] = 0;
    v2[10] = 0;
    v2[12] = 0;
    v2[13] = 0;
    v2[15] = 1;
    v2[16] = 0;
    v2[17] = 0;
    v2[19] = 0;
    v2[20] = 0;
    *((_BYTE *)v2 + 168) = 0;
    v2[22] = 0;
    v2[23] = 0;
    *((_DWORD *)v2 + 22) = 1065353216;
    *((_DWORD *)v2 + 36) = 1065353216;
    if ( v3 )
    {
      sub_C92620((__int64)(v2 + 22), *((_DWORD *)a1 + 2));
      v6 = v2[22];
      v7 = *a1;
      v8 = *((unsigned int *)v2 + 46);
      v9 = 8 * v8 + 8;
      v20 = v6;
      v19 = *a1;
      *(_QWORD *)((char *)v2 + 188) = *(__int64 *)((char *)a1 + 12);
      if ( (_DWORD)v8 )
      {
        v10 = 0;
        v21 = 8LL * (unsigned int)(v8 - 1);
        v11 = v7;
        while ( 1 )
        {
          v16 = *(size_t **)(v11 + v10);
          v17 = (size_t **)(v6 + v10);
          if ( v16 == (size_t *)-8LL || !v16 )
          {
            *v17 = v16;
          }
          else
          {
            v22 = *(_QWORD *)(v11 + v10);
            n = *v16;
            v12 = sub_C7D670(*v16 + 17, 8);
            v13 = n;
            v14 = v22;
            v15 = v12;
            if ( n )
            {
              v18 = v12;
              memcpy((void *)(v12 + 16), (const void *)(v22 + 16), n);
              v13 = n;
              v14 = v22;
              v15 = v18;
            }
            *(_BYTE *)(v15 + v13 + 16) = 0;
            *(_QWORD *)v15 = v13;
            *(_DWORD *)(v15 + 8) = *(_DWORD *)(v14 + 8);
            *v17 = (size_t *)v15;
            *(_DWORD *)(v20 + v9) = *(_DWORD *)(v19 + v9);
          }
          v9 += 4;
          if ( v10 == v21 )
            break;
          v11 = *a1;
          v6 = v2[22];
          v10 += 8;
        }
      }
    }
    v4 = sub_BC2B00();
    sub_2D02860((__int64)v4);
  }
  return v2;
}
