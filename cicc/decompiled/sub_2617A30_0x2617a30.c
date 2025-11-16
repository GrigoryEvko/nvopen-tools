// Function: sub_2617A30
// Address: 0x2617a30
//
__int64 __fastcall sub_2617A30(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v6; // r14
  __int64 v7; // rsi
  __int64 v8; // rdx
  unsigned int v9; // r12d
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v12; // r13
  __int64 v15; // [rsp+10h] [rbp-230h]
  unsigned __int64 v16[2]; // [rsp+20h] [rbp-220h] BYREF
  _BYTE v17[16]; // [rsp+30h] [rbp-210h] BYREF
  unsigned __int64 v18[2]; // [rsp+40h] [rbp-200h] BYREF
  char v19; // [rsp+50h] [rbp-1F0h] BYREF
  __int64 v20; // [rsp+D8h] [rbp-168h]
  unsigned int v21; // [rsp+E8h] [rbp-158h]
  __int64 v22; // [rsp+F8h] [rbp-148h]
  unsigned int v23; // [rsp+108h] [rbp-138h]
  _BYTE v24[64]; // [rsp+110h] [rbp-130h] BYREF
  __int64 v25; // [rsp+150h] [rbp-F0h]
  unsigned int v26; // [rsp+160h] [rbp-E0h]
  unsigned __int64 *v27; // [rsp+168h] [rbp-D8h]
  char *v28; // [rsp+178h] [rbp-C8h] BYREF
  char v29; // [rsp+188h] [rbp-B8h] BYREF
  _QWORD *v30; // [rsp+1B8h] [rbp-88h]
  _QWORD v31[6]; // [rsp+1C8h] [rbp-78h] BYREF
  unsigned int v32; // [rsp+1F8h] [rbp-48h]
  char *v33; // [rsp+200h] [rbp-40h]
  char v34; // [rsp+210h] [rbp-30h] BYREF

  v6 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 72LL);
  v15 = (*(__int64 (__fastcall **)(_QWORD, __int64))(a1 + 40))(*(_QWORD *)(a1 + 48), v6);
  sub_29B4290(v18, v6);
  v7 = *(_QWORD *)(a2 + 32);
  v8 = (*(_QWORD *)(a2 + 40) - v7) >> 3;
  v16[0] = (unsigned __int64)v17;
  v16[1] = 0;
  v17[0] = 0;
  sub_29AFB10((unsigned int)v24, v7, v8, a4, 0, 0, 0, v15, 0, 0, 0, (__int64)v16, 0);
  if ( (_BYTE *)v16[0] != v17 )
    j_j___libc_free_0(v16[0]);
  v9 = 0;
  if ( sub_29B77F0(v24, v18) )
  {
    v9 = 1;
    sub_D4F720(a3, (__int64 *)a2);
    --*(_DWORD *)a1;
  }
  if ( v33 != &v34 )
    _libc_free((unsigned __int64)v33);
  sub_C7D6A0(v31[4], 8LL * v32, 8);
  if ( v30 != v31 )
    j_j___libc_free_0((unsigned __int64)v30);
  if ( v28 != &v29 )
    _libc_free((unsigned __int64)v28);
  if ( v27 != (unsigned __int64 *)&v28 )
    _libc_free((unsigned __int64)v27);
  sub_C7D6A0(v25, 8LL * v26, 8);
  sub_C7D6A0(v22, 8LL * v23, 8);
  v10 = v21;
  if ( v21 )
  {
    v11 = v20;
    v12 = v20 + 40LL * v21;
    do
    {
      if ( *(_QWORD *)v11 != -4096 && *(_QWORD *)v11 != -8192 )
        sub_C7D6A0(*(_QWORD *)(v11 + 16), 8LL * *(unsigned int *)(v11 + 32), 8);
      v11 += 40;
    }
    while ( v12 != v11 );
    v10 = v21;
  }
  sub_C7D6A0(v20, 40 * v10, 8);
  if ( (char *)v18[0] != &v19 )
    _libc_free(v18[0]);
  return v9;
}
