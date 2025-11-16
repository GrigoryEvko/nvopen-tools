// Function: sub_FB95C0
// Address: 0xfb95c0
//
__int64 __fastcall sub_FB95C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  _BYTE *v4; // rbx
  _BYTE *v5; // r13
  __int64 v6; // r14
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r15
  unsigned int v13; // eax
  unsigned int v14; // r9d
  unsigned int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rax
  unsigned int *v18; // rbx
  __int64 v19; // rdx
  unsigned int v20; // [rsp+Ch] [rbp-A4h]
  __int64 v21; // [rsp+10h] [rbp-A0h]
  __int64 v22; // [rsp+18h] [rbp-98h]
  __int64 v23; // [rsp+20h] [rbp-90h]
  unsigned __int8 v24; // [rsp+28h] [rbp-88h]
  unsigned int *v25; // [rsp+30h] [rbp-80h] BYREF
  __int64 v26; // [rsp+38h] [rbp-78h]
  _BYTE v27[112]; // [rsp+40h] [rbp-70h] BYREF

  result = 0;
  v4 = *(_BYTE **)(a3 - 64);
  v5 = *(_BYTE **)(a3 - 32);
  if ( *v4 == 17 && *v5 == 17 )
  {
    v23 = *(_QWORD *)(a3 - 96);
    v6 = ((*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1) - 1;
    sub_F90F90(a2, 0, a2, v6, (__int64)v4);
    if ( v6 == v7 || (_DWORD)v7 == -2 )
      v8 = 32;
    else
      v8 = 32LL * (unsigned int)(2 * v7 + 3);
    v9 = *(_QWORD *)(a2 - 8);
    v22 = *(_QWORD *)(v9 + v8);
    sub_F90F90(a2, 0, a2, v6, (__int64)v5);
    v11 = 32;
    if ( v6 != v10 && (_DWORD)v10 != -2 )
      v11 = 32LL * (unsigned int)(2 * v10 + 3);
    v12 = *(_QWORD *)(v9 + v11);
    v25 = (unsigned int *)v27;
    v26 = 0x800000000LL;
    if ( (unsigned __int8)sub_BC8700(a2)
      && (sub_F8F540((_BYTE *)a2, (__int64)&v25), v15 = (*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1, (_DWORD)v26 == v15) )
    {
      v21 = v15 - 1;
      sub_F90F90(a2, 0, a2, v21, (__int64)v4);
      v17 = 0;
      if ( v16 != v21 && (_DWORD)v16 != -2 )
        v17 = 2LL * (unsigned int)(v16 + 1);
      v18 = v25;
      v20 = v25[v17];
      sub_F90F90(a2, 0, a2, v21, (__int64)v5);
      v14 = v20;
      if ( v21 != v19 && (_DWORD)v19 != -2 )
        v18 += 2 * (unsigned int)(v19 + 1);
      v13 = *v18;
    }
    else
    {
      v13 = 0;
      v14 = 0;
    }
    result = sub_FB8CA0(a1, a2, v23, v22, v12, v14, v13);
    if ( v25 != (unsigned int *)v27 )
    {
      v24 = result;
      _libc_free(v25, a2);
      return v24;
    }
  }
  return result;
}
