// Function: sub_2E0A330
// Address: 0x2e0a330
//
void __fastcall sub_2E0A330(__int64 *a1, __int64 a2, char *a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // r12
  char v9; // al
  __int64 v10; // rax
  _DWORD *v11; // rbx
  char *v12; // rax
  __int64 v13; // [rsp+0h] [rbp-90h] BYREF
  char *v14; // [rsp+8h] [rbp-88h]
  __int64 v15; // [rsp+10h] [rbp-80h]
  int v16; // [rsp+18h] [rbp-78h]
  char v17; // [rsp+1Ch] [rbp-74h]
  char v18; // [rsp+20h] [rbp-70h] BYREF

  v14 = &v18;
  v6 = *((unsigned int *)a1 + 2);
  v7 = *a1;
  v15 = 8;
  v16 = 0;
  v8 = v7 + 24 * v6;
  v17 = 1;
  v13 = 0;
  *((_DWORD *)a1 + 18) = 0;
  if ( v8 == v7 )
    return;
  v9 = 1;
  while ( 1 )
  {
    v11 = *(_DWORD **)(v7 + 16);
    if ( !v9 )
      goto LABEL_3;
    a4 = (unsigned __int64)v14;
    a3 = &v14[8 * HIDWORD(v15)];
    v12 = v14;
    if ( v14 != a3 )
      break;
LABEL_15:
    if ( HIDWORD(v15) < (unsigned int)v15 )
    {
      ++HIDWORD(v15);
      *(_QWORD *)a3 = v11;
      ++v13;
LABEL_4:
      *v11 = *((_DWORD *)a1 + 18);
      v10 = *((unsigned int *)a1 + 18);
      a4 = *((unsigned int *)a1 + 19);
      if ( v10 + 1 > a4 )
      {
        sub_C8D5F0((__int64)(a1 + 8), a1 + 10, v10 + 1, 8u, a5, a6);
        v10 = *((unsigned int *)a1 + 18);
      }
      a3 = (char *)a1[8];
      *(_QWORD *)&a3[8 * v10] = v11;
      v9 = v17;
      ++*((_DWORD *)a1 + 18);
      goto LABEL_7;
    }
LABEL_3:
    sub_C8CC70((__int64)&v13, (__int64)v11, (__int64)a3, a4, a5, a6);
    v9 = v17;
    if ( (_BYTE)a3 )
      goto LABEL_4;
LABEL_7:
    v7 += 24;
    if ( v7 == v8 )
    {
      if ( !v9 )
        _libc_free((unsigned __int64)v14);
      return;
    }
  }
  while ( 1 )
  {
    while ( *(_DWORD **)v12 != v11 )
    {
      v12 += 8;
      if ( a3 == v12 )
        goto LABEL_15;
    }
    v7 += 24;
    if ( v8 == v7 )
      break;
    v11 = *(_DWORD **)(v7 + 16);
    v12 = v14;
    if ( v14 == a3 )
      goto LABEL_15;
  }
}
