// Function: sub_18160E0
// Address: 0x18160e0
//
__int64 __fastcall sub_18160E0(__int64 a1, __int64 a2)
{
  __int64 v3; // r9
  __int64 v4; // r8
  __int64 v5; // rdx
  const void *v6; // rax
  __int64 v7; // r8
  const void *v8; // r9
  __int64 v9; // rbx
  int v10; // eax
  __int64 v11; // r8
  int v12; // ecx
  __int64 v13; // rdx
  unsigned __int64 v14; // r9
  int v15; // ebx
  _QWORD *v16; // rax
  _QWORD *v17; // rdx
  __int64 **v18; // rax
  __int64 v19; // rbx
  __int64 *v20; // rdi
  __int64 v21; // r12
  __int64 v23; // r12
  __int64 **v24; // rax
  __int64 v25; // [rsp+8h] [rbp-78h]
  const void *v26; // [rsp+10h] [rbp-70h]
  __int64 v27; // [rsp+10h] [rbp-70h]
  const void *v28; // [rsp+18h] [rbp-68h]
  unsigned __int64 v29; // [rsp+18h] [rbp-68h]
  _QWORD *v30; // [rsp+20h] [rbp-60h] BYREF
  __int64 v31; // [rsp+28h] [rbp-58h]
  _QWORD v32[10]; // [rsp+30h] [rbp-50h] BYREF

  v30 = v32;
  v31 = 0x400000000LL;
  v32[0] = sub_1647190((__int64 *)a2, 0);
  v3 = *(_QWORD *)(a2 + 16);
  v4 = 8LL * *(unsigned int *)(a2 + 12);
  v6 = (const void *)(v3 + v4);
  v7 = v4 - 8;
  v8 = (const void *)(v3 + 8);
  LODWORD(v31) = 1;
  v5 = 1;
  v9 = v7 >> 3;
  if ( (unsigned __int64)(v7 >> 3) > 3 )
  {
    v25 = v7;
    v26 = v6;
    v28 = v8;
    sub_16CD150((__int64)&v30, v32, v9 + (unsigned int)v31, 8, v7, (int)v8);
    v5 = (unsigned int)v31;
    v7 = v25;
    v6 = v26;
    v8 = v28;
  }
  if ( v8 != v6 )
  {
    memcpy(&v30[v5], v8, v7);
    LODWORD(v5) = v31;
  }
  v10 = *(_DWORD *)(a2 + 12);
  v11 = *(_QWORD *)(a1 + 176);
  LODWORD(v31) = v9 + v5;
  v12 = v9 + v5;
  v13 = (unsigned int)(v9 + v5);
  v14 = (unsigned int)(v10 - 1);
  v15 = v10 - 1;
  if ( v14 > (unsigned __int64)HIDWORD(v31) - v13 )
  {
    v27 = v11;
    v29 = (unsigned int)(v10 - 1);
    sub_16CD150((__int64)&v30, v32, v14 + v13, 8, v11, v14);
    v13 = (unsigned int)v31;
    v11 = v27;
    v14 = v29;
    v12 = v31;
  }
  if ( v14 )
  {
    v16 = &v30[v13];
    v17 = &v16[v14];
    do
      *v16++ = v11;
    while ( v17 != v16 );
    v12 = v31;
  }
  v18 = *(__int64 ***)(a2 + 16);
  v19 = (unsigned int)(v12 + v15);
  LODWORD(v31) = v19;
  v20 = *v18;
  if ( *((_BYTE *)*v18 + 8) )
  {
    v23 = *(_QWORD *)(a1 + 184);
    if ( (unsigned int)v19 >= HIDWORD(v31) )
    {
      sub_16CD150((__int64)&v30, v32, 0, 8, v11, v14);
      v19 = (unsigned int)v31;
    }
    v30[v19] = v23;
    LODWORD(v19) = v31 + 1;
    v24 = *(__int64 ***)(a2 + 16);
    LODWORD(v31) = v31 + 1;
    v20 = *v24;
  }
  v21 = sub_1644EA0(v20, v30, (unsigned int)v19, 0);
  if ( v30 != v32 )
    _libc_free((unsigned __int64)v30);
  return v21;
}
