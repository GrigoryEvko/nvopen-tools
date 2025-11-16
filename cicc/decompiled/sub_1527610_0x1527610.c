// Function: sub_1527610
// Address: 0x1527610
//
__int64 __fastcall sub_1527610(_QWORD **a1)
{
  _QWORD *v1; // rax
  _QWORD *v2; // rbx
  __int64 v3; // r13
  __int64 v4; // rax
  _QWORD *v5; // rax
  __int64 v6; // rax
  _QWORD *v7; // rax
  __int64 v8; // rax
  _QWORD *v9; // rax
  __int64 v10; // rax
  _QWORD *v11; // rax
  __int64 v12; // rax
  _QWORD *v13; // rax
  __int64 v14; // rax
  _QWORD *v15; // rax
  _QWORD *v16; // rdi
  unsigned int v17; // r12d
  __int64 v19; // [rsp+0h] [rbp-40h] BYREF
  volatile signed __int32 *v20; // [rsp+8h] [rbp-38h]

  v1 = (_QWORD *)sub_22077B0(544);
  v2 = v1;
  if ( v1 )
  {
    v3 = (__int64)(v1 + 2);
    v1[1] = 0x100000001LL;
    *v1 = &unk_49ECD20;
    v1[2] = v1 + 4;
    v1[3] = 0x2000000000LL;
    v4 = 0;
  }
  else
  {
    v4 = MEMORY[0x18];
    v3 = 16;
    if ( MEMORY[0x18] >= MEMORY[0x1C] )
    {
      sub_16CD150(16, 32, 0, 16);
      v4 = MEMORY[0x18];
    }
  }
  v5 = (_QWORD *)(v2[2] + 16 * v4);
  *v5 = 7;
  v5[1] = 1;
  v6 = (unsigned int)(*((_DWORD *)v2 + 6) + 1);
  *((_DWORD *)v2 + 6) = v6;
  if ( *((_DWORD *)v2 + 7) <= (unsigned int)v6 )
  {
    sub_16CD150(v3, v2 + 4, 0, 16);
    v6 = *((unsigned int *)v2 + 6);
  }
  v7 = (_QWORD *)(v2[2] + 16 * v6);
  *v7 = 1;
  v7[1] = 2;
  v8 = (unsigned int)(*((_DWORD *)v2 + 6) + 1);
  *((_DWORD *)v2 + 6) = v8;
  if ( *((_DWORD *)v2 + 7) <= (unsigned int)v8 )
  {
    sub_16CD150(v3, v2 + 4, 0, 16);
    v8 = *((unsigned int *)v2 + 6);
  }
  v9 = (_QWORD *)(v2[2] + 16 * v8);
  *v9 = 6;
  v9[1] = 4;
  v10 = (unsigned int)(*((_DWORD *)v2 + 6) + 1);
  *((_DWORD *)v2 + 6) = v10;
  if ( *((_DWORD *)v2 + 7) <= (unsigned int)v10 )
  {
    sub_16CD150(v3, v2 + 4, 0, 16);
    v10 = *((unsigned int *)v2 + 6);
  }
  v11 = (_QWORD *)(v2[2] + 16 * v10);
  *v11 = 8;
  v11[1] = 4;
  v12 = (unsigned int)(*((_DWORD *)v2 + 6) + 1);
  *((_DWORD *)v2 + 6) = v12;
  if ( *((_DWORD *)v2 + 7) <= (unsigned int)v12 )
  {
    sub_16CD150(v3, v2 + 4, 0, 16);
    v12 = *((unsigned int *)v2 + 6);
  }
  v13 = (_QWORD *)(v2[2] + 16 * v12);
  *v13 = 6;
  v13[1] = 4;
  v14 = (unsigned int)(*((_DWORD *)v2 + 6) + 1);
  *((_DWORD *)v2 + 6) = v14;
  if ( *((_DWORD *)v2 + 7) <= (unsigned int)v14 )
  {
    sub_16CD150(v3, v2 + 4, 0, 16);
    v14 = *((unsigned int *)v2 + 6);
  }
  v15 = (_QWORD *)(v2[2] + 16 * v14);
  *v15 = 6;
  v15[1] = 4;
  v16 = *a1;
  ++*((_DWORD *)v2 + 6);
  v19 = v3;
  v20 = (volatile signed __int32 *)v2;
  v17 = sub_15271D0(v16, &v19);
  if ( v20 )
    sub_A191D0(v20);
  return v17;
}
