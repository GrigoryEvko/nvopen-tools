// Function: sub_3504900
// Address: 0x3504900
//
void __fastcall sub_3504900(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rdi
  int v7; // ebx
  __int64 v8; // rax
  int v9; // eax
  __int64 *v10; // rax
  unsigned __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // r15
  unsigned __int64 v16; // rdx
  _QWORD *v17; // rax
  _QWORD *v18; // [rsp+0h] [rbp-80h] BYREF
  __int64 v19; // [rsp+8h] [rbp-78h]
  _QWORD v20[14]; // [rsp+10h] [rbp-70h] BYREF

  v6 = v20;
  v7 = 0;
  v19 = 0x400000001LL;
  v8 = 1;
  v18 = v20;
  v20[0] = a2;
  v20[1] = 0;
  do
  {
    while ( 1 )
    {
      ++v7;
      v10 = &v6[2 * v8 - 2];
      v11 = v10[1];
      v12 = *v10;
      v10[1] = v11 + 1;
      if ( v11 < *(unsigned int *)(v12 + 40) )
        break;
      v9 = v19;
      *(_DWORD *)(v12 + 180) = v7;
      v8 = (unsigned int)(v9 - 1);
      LODWORD(v19) = v8;
      if ( !(_DWORD)v8 )
        goto LABEL_7;
    }
    v13 = *(_QWORD *)(v12 + 32) + 8 * v11;
    v14 = (unsigned int)v19;
    v15 = *(_QWORD *)v13;
    v16 = (unsigned int)v19 + 1LL;
    if ( v16 > HIDWORD(v19) )
    {
      sub_C8D5F0((__int64)&v18, v20, v16, 0x10u, a5, a6);
      v6 = v18;
      v14 = (unsigned int)v19;
    }
    v17 = &v6[2 * v14];
    *v17 = v15;
    v6 = v18;
    v17[1] = 0;
    LODWORD(v19) = v19 + 1;
    v8 = (unsigned int)v19;
    *(_DWORD *)(*(_QWORD *)v13 + 176LL) = v7;
  }
  while ( (_DWORD)v8 );
LABEL_7:
  if ( v6 != v20 )
    _libc_free((unsigned __int64)v6);
}
