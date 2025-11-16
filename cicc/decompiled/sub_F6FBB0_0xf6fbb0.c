// Function: sub_F6FBB0
// Address: 0xf6fbb0
//
__int64 __fastcall sub_F6FBB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v6; // r13
  __int64 v7; // rax
  _BYTE *v8; // rdx
  _QWORD *v9; // r15
  unsigned int *v10; // r14
  __int64 v11; // rbx
  __int64 v12; // rdx
  unsigned int v13; // esi
  __int64 v15; // [rsp+8h] [rbp-98h]
  _BYTE v16[32]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v17; // [rsp+30h] [rbp-70h]
  unsigned __int64 v18[2]; // [rsp+40h] [rbp-60h] BYREF
  __int64 v19; // [rsp+50h] [rbp-50h]
  __int16 v20; // [rsp+60h] [rbp-40h]

  v4 = a2;
  v6 = *(_QWORD *)(a3 + 24);
  v18[0] = 6;
  v18[1] = 0;
  v19 = v6;
  if ( v6 != -4096 && v6 != 0 && v6 != -8192 )
  {
    sub_BD6050(v18, *(_QWORD *)(a3 + 8) & 0xFFFFFFFFFFFFFFF8LL);
    v6 = v19;
    if ( v19 != -4096 && v19 != 0 && v19 != -8192 )
      sub_BD60C0(v18);
  }
  v7 = *(_QWORD *)(a4 + 16);
  if ( !v7 )
LABEL_19:
    BUG();
  while ( 1 )
  {
    v8 = *(_BYTE **)(v7 + 24);
    if ( *v8 == 86 )
      break;
    v7 = *(_QWORD *)(v7 + 8);
    if ( !v7 )
      goto LABEL_19;
  }
  v15 = *((_QWORD *)v8 - 8);
  if ( a4 == v15 )
    v15 = *((_QWORD *)v8 - 4);
  if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a2 + 8) + 8LL) - 17 <= 1 )
    v4 = sub_B34870(a1, a2);
  v17 = 257;
  v20 = 257;
  v9 = sub_BD2C40(72, unk_3F10A14);
  if ( v9 )
    sub_B549F0((__int64)v9, v4, (__int64)v18, 0, 0);
  (*(void (__fastcall **)(_QWORD, _QWORD *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
    *(_QWORD *)(a1 + 88),
    v9,
    v16,
    *(_QWORD *)(a1 + 56),
    *(_QWORD *)(a1 + 64));
  v10 = *(unsigned int **)a1;
  v11 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v11 )
  {
    do
    {
      v12 = *((_QWORD *)v10 + 1);
      v13 = *v10;
      v10 += 4;
      sub_B99FD0((__int64)v9, v13, v12);
    }
    while ( (unsigned int *)v11 != v10 );
  }
  v18[0] = (unsigned __int64)"rdx.select";
  v20 = 259;
  return sub_B36550((unsigned int **)a1, (__int64)v9, v15, v6, (__int64)v18, 0);
}
