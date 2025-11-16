// Function: sub_1E1C690
// Address: 0x1e1c690
//
__int64 __fastcall sub_1E1C690(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // r12
  _QWORD *v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r12
  int v14; // r8d
  int v15; // r9d
  __int64 v16; // r15
  __int64 v17; // rax
  unsigned int v18; // ebx
  int v20; // r8d
  int v21; // r9d
  _QWORD *v22; // [rsp+0h] [rbp-80h] BYREF
  __int64 v23; // [rsp+8h] [rbp-78h]
  _BYTE v24[112]; // [rsp+10h] [rbp-70h] BYREF

  v23 = 0x800000000LL;
  v6 = *a1;
  v22 = v24;
  v7 = (unsigned int)(*(_DWORD *)(v6 + 40) + 1);
  if ( v7 > 8 )
  {
    sub_16CD150((__int64)&v22, v24, v7, 8, a5, a6);
    v8 = **(unsigned __int16 **)(*a1 + 16);
    if ( (unsigned int)v23 >= HIDWORD(v23) )
      sub_16CD150((__int64)&v22, v24, 0, 8, v20, v21);
    v9 = &v22[(unsigned int)v23];
  }
  else
  {
    v8 = **(unsigned __int16 **)(v6 + 16);
    v9 = v24;
  }
  *v9 = v8;
  v10 = *(_QWORD *)(*a1 + 32);
  v11 = (unsigned int)(v23 + 1);
  v12 = 5LL * *(unsigned int *)(*a1 + 40);
  LODWORD(v23) = v23 + 1;
  v13 = v10 + 8 * v12;
  if ( v10 != v13 )
  {
    do
    {
      if ( *(_BYTE *)v10 || (*(_BYTE *)(v10 + 3) & 0x10) == 0 || *(int *)(v10 + 8) >= 0 )
      {
        v16 = sub_1E36300(v10);
        v17 = (unsigned int)v23;
        if ( (unsigned int)v23 >= HIDWORD(v23) )
        {
          sub_16CD150((__int64)&v22, v24, 0, 8, v14, v15);
          v17 = (unsigned int)v23;
        }
        v22[v17] = v16;
        LODWORD(v23) = v23 + 1;
      }
      v10 += 40;
    }
    while ( v13 != v10 );
    v11 = (unsigned int)v23;
  }
  v18 = sub_16AF040(v22, (__int64)&v22[v11]);
  if ( v22 != (_QWORD *)v24 )
    _libc_free((unsigned __int64)v22);
  return v18;
}
