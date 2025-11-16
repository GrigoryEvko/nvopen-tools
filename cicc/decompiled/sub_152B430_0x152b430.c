// Function: sub_152B430
// Address: 0x152b430
//
__int64 __fastcall sub_152B430(__int64 a1, unsigned int a2, unsigned int a3, unsigned __int8 *a4, __int64 a5)
{
  __int64 v8; // r12
  _QWORD *v9; // rax
  _QWORD *v10; // rbx
  __int64 v11; // r8
  __int64 v12; // rax
  _QWORD *v13; // rax
  __int64 v14; // rax
  _QWORD *v15; // rax
  _QWORD *v16; // rdi
  unsigned int v17; // eax
  unsigned int v18; // esi
  __int64 v19; // rdi
  unsigned int v21; // [rsp+8h] [rbp-58h]
  __int64 v22; // [rsp+8h] [rbp-58h]
  __int64 v23; // [rsp+18h] [rbp-48h] BYREF
  __int64 v24; // [rsp+20h] [rbp-40h] BYREF
  volatile signed __int32 *v25; // [rsp+28h] [rbp-38h]

  v8 = a3;
  sub_1526BE0(*(_QWORD **)(a1 + 8), a2, 3u);
  v9 = (_QWORD *)sub_22077B0(544);
  v10 = v9;
  if ( v9 )
  {
    v11 = (__int64)(v9 + 2);
    v9[1] = 0x100000001LL;
    *v9 = &unk_49ECD20;
    v9[2] = v9 + 4;
    v9[3] = 0x2000000000LL;
    v12 = 0;
  }
  else
  {
    v12 = MEMORY[0x18];
    v11 = 16;
    if ( MEMORY[0x1C] <= MEMORY[0x18] )
    {
      sub_16CD150(16, 32, 0, 16);
      v12 = MEMORY[0x18];
      v11 = 16;
    }
  }
  v13 = (_QWORD *)(v10[2] + 16 * v12);
  *v13 = v8;
  v13[1] = 1;
  v14 = (unsigned int)(*((_DWORD *)v10 + 6) + 1);
  *((_DWORD *)v10 + 6) = v14;
  if ( *((_DWORD *)v10 + 7) <= (unsigned int)v14 )
  {
    v22 = v11;
    sub_16CD150(v11, v10 + 4, 0, 16);
    v14 = *((unsigned int *)v10 + 6);
    v11 = v22;
  }
  v15 = (_QWORD *)(v10[2] + 16 * v14);
  *v15 = 0;
  v15[1] = 10;
  v16 = *(_QWORD **)(a1 + 8);
  ++*((_DWORD *)v10 + 6);
  v25 = (volatile signed __int32 *)v10;
  v24 = v11;
  v17 = sub_15271D0(v16, &v24);
  v18 = v17;
  if ( v25 )
  {
    v21 = v17;
    sub_A191D0(v25);
    v18 = v21;
  }
  v19 = *(_QWORD *)(a1 + 8);
  v23 = v8;
  BYTE4(v24) = 0;
  sub_152A250(v19, v18, (__int64)&v23, 1, a4, a5, (__int64)&v24);
  return sub_15263C0(*(__int64 ***)(a1 + 8));
}
