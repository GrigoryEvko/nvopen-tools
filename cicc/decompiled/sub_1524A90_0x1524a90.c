// Function: sub_1524A90
// Address: 0x1524a90
//
void __fastcall sub_1524A90(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int *v5; // r13
  unsigned int *v6; // r15
  __int64 v7; // rax
  __int64 v8; // r9
  __int64 v9; // r9
  __int64 v10; // rax
  int v11; // ecx
  __int64 v12; // rdi
  int v13; // ecx
  __int64 v14; // r9
  unsigned int v15; // esi
  __int64 *v16; // rdx
  __int64 v17; // r11
  int v18; // edx
  int v19; // r10d
  __int64 v20; // [rsp+0h] [rbp-90h]
  __int64 v21; // [rsp+0h] [rbp-90h]
  unsigned int *v22; // [rsp+10h] [rbp-80h] BYREF
  __int64 v23; // [rsp+18h] [rbp-78h]
  _BYTE v24[112]; // [rsp+20h] [rbp-70h] BYREF

  v22 = (unsigned int *)v24;
  v23 = 0x400000000LL;
  sub_1626D60(a3, &v22);
  v5 = v22;
  v6 = &v22[4 * (unsigned int)v23];
  if ( v6 != v22 )
  {
    v7 = *(unsigned int *)(a2 + 8);
    do
    {
      v8 = *v5;
      if ( (unsigned int)v7 >= *(_DWORD *)(a2 + 12) )
      {
        v21 = *v5;
        sub_16CD150(a2, a2 + 16, 0, 8);
        v7 = *(unsigned int *)(a2 + 8);
        v8 = v21;
      }
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v7) = v8;
      v9 = 0xFFFFFFFFLL;
      v10 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
      *(_DWORD *)(a2 + 8) = v10;
      v11 = *(_DWORD *)(a1 + 304);
      if ( v11 )
      {
        v12 = *((_QWORD *)v5 + 1);
        v13 = v11 - 1;
        v14 = *(_QWORD *)(a1 + 288);
        v15 = v13 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v16 = (__int64 *)(v14 + 16LL * v15);
        v17 = *v16;
        if ( v12 == *v16 )
        {
LABEL_7:
          v9 = (unsigned int)(*((_DWORD *)v16 + 3) - 1);
        }
        else
        {
          v18 = 1;
          while ( v17 != -4 )
          {
            v19 = v18 + 1;
            v15 = v13 & (v18 + v15);
            v16 = (__int64 *)(v14 + 16LL * v15);
            v17 = *v16;
            if ( v12 == *v16 )
              goto LABEL_7;
            v18 = v19;
          }
          v9 = 0xFFFFFFFFLL;
        }
      }
      if ( *(_DWORD *)(a2 + 12) <= (unsigned int)v10 )
      {
        v20 = v9;
        sub_16CD150(a2, a2 + 16, 0, 8);
        v10 = *(unsigned int *)(a2 + 8);
        v9 = v20;
      }
      v5 += 4;
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v10) = v9;
      v7 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
      *(_DWORD *)(a2 + 8) = v7;
    }
    while ( v6 != v5 );
    v5 = v22;
  }
  if ( v5 != (unsigned int *)v24 )
    _libc_free((unsigned __int64)v5);
}
