// Function: sub_1994A60
// Address: 0x1994a60
//
void __fastcall sub_1994A60(__int64 a1, __int64 *a2)
{
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rcx
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // r9
  __int64 v9; // rdx
  __int64 v10; // rcx
  int v11; // r9d
  char *v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned __int64 v15; // rdi
  __int64 v16; // [rsp+10h] [rbp-90h]
  __int64 v17; // [rsp+18h] [rbp-88h]
  char v18; // [rsp+20h] [rbp-80h]
  __int64 v19; // [rsp+28h] [rbp-78h]
  char *v20[2]; // [rsp+30h] [rbp-70h] BYREF
  _BYTE v21[32]; // [rsp+40h] [rbp-60h] BYREF
  __int64 v22; // [rsp+60h] [rbp-40h]
  __int64 v23; // [rsp+68h] [rbp-38h]

  v3 = *(_QWORD *)(a1 + 744);
  v4 = *(unsigned int *)(a1 + 752);
  v5 = 96 * v4;
  v6 = v3 + 96 * v4 - 96;
  if ( a2 != (__int64 *)v6 )
  {
    v7 = *a2;
    v8 = (__int64)(a2 + 4);
    v20[0] = v21;
    v16 = v7;
    v17 = a2[1];
    v18 = *((_BYTE *)a2 + 16);
    v19 = a2[3];
    v20[1] = (char *)0x400000000LL;
    if ( *((_DWORD *)a2 + 10) )
    {
      sub_19931B0((__int64)v20, (char **)a2 + 4, v3, v5, (int)v20, v8);
      v8 = (__int64)(a2 + 4);
    }
    v22 = a2[10];
    v23 = a2[11];
    *a2 = *(_QWORD *)v6;
    a2[1] = *(_QWORD *)(v6 + 8);
    *((_BYTE *)a2 + 16) = *(_BYTE *)(v6 + 16);
    a2[3] = *(_QWORD *)(v6 + 24);
    sub_19931B0(v8, (char **)(v6 + 32), v3, v5, (int)v20, v8);
    a2[10] = *(_QWORD *)(v6 + 80);
    a2[11] = *(_QWORD *)(v6 + 88);
    *(_QWORD *)v6 = v16;
    *(_QWORD *)(v6 + 8) = v17;
    *(_BYTE *)(v6 + 16) = v18;
    *(_QWORD *)(v6 + 24) = v19;
    sub_19931B0(v6 + 32, v20, v9, v10, (int)v20, v11);
    v12 = v20[0];
    *(_QWORD *)(v6 + 80) = v22;
    *(_QWORD *)(v6 + 88) = v23;
    if ( v12 != v21 )
      _libc_free((unsigned __int64)v12);
    LODWORD(v4) = *(_DWORD *)(a1 + 752);
    v3 = *(_QWORD *)(a1 + 744);
  }
  v13 = (unsigned int)(v4 - 1);
  *(_DWORD *)(a1 + 752) = v13;
  v14 = 96 * v13 + v3;
  v15 = *(_QWORD *)(v14 + 32);
  if ( v15 != v14 + 48 )
    _libc_free(v15);
}
