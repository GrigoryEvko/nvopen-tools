// Function: sub_1B965E0
// Address: 0x1b965e0
//
void __fastcall sub_1B965E0(__int128 a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // r8
  _QWORD *v6; // r15
  _QWORD *v7; // rbx
  char v8; // al
  int v9; // r9d
  __int64 v10; // r8
  char v11; // al
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // [rsp+8h] [rbp-98h]
  __int64 v15; // [rsp+8h] [rbp-98h]
  char v16[8]; // [rsp+10h] [rbp-90h] BYREF
  int v17; // [rsp+18h] [rbp-88h]
  __int64 v18; // [rsp+20h] [rbp-80h]
  unsigned int v19; // [rsp+28h] [rbp-78h]
  __int64 v20; // [rsp+38h] [rbp-68h]
  __int64 v21; // [rsp+48h] [rbp-58h]

  v3 = a1;
  v6 = *(_QWORD **)(a1 + 16);
  v7 = *(_QWORD **)(a1 + 8);
  if ( v7 != v6 && !byte_4FB7F60 )
  {
    if ( !byte_4FB8040 )
    {
      do
      {
LABEL_8:
        *(_QWORD *)&a1 = *v7++;
        sub_1B965E0(a1, *((_QWORD *)&a1 + 1), a2, a3);
      }
      while ( v6 != v7 );
      return;
    }
    sub_1BF1BF0(v16, a1, 1, a2, a1);
    v3 = a1;
    if ( (_DWORD)v20 != -1 )
    {
      v11 = sub_1BF5810(v16, *(_QWORD *)(**(_QWORD **)(a1 + 32) + 56LL), a1, 0);
      v3 = a1;
      if ( v11 )
      {
        if ( v17 && v19 <= 1 )
          goto LABEL_3;
        sub_1BF5800(v16);
        v3 = a1;
        if ( (_DWORD)v20 == 1 )
        {
          sub_1B955D0(a1, (__int64)v16, a2);
          v3 = a1;
        }
      }
    }
LABEL_7:
    v7 = *(_QWORD **)(v3 + 8);
    v6 = *(_QWORD **)(v3 + 16);
    if ( v6 == v7 )
      return;
    goto LABEL_8;
  }
LABEL_3:
  v14 = v3;
  sub_1AFCDB0((__int64)v16, v3);
  *(_QWORD *)&a1 = v16;
  sub_13FF3D0(a1);
  v8 = sub_1A523B0((__int64)v16, *((__int64 *)&a1 + 1));
  v10 = v14;
  if ( v8 )
  {
    if ( v20 )
    {
      j_j___libc_free_0(v20, v21 - v20);
      v10 = v14;
    }
    v15 = v10;
    j___libc_free_0(v18);
    v3 = v15;
    goto LABEL_7;
  }
  v12 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v12 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, (const void *)(a3 + 16), 0, 8, v14, v9);
    v12 = *(unsigned int *)(a3 + 8);
    v10 = v14;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v12) = v10;
  v13 = v20;
  ++*(_DWORD *)(a3 + 8);
  if ( v13 )
    j_j___libc_free_0(v13, v21 - v13);
  j___libc_free_0(v18);
}
