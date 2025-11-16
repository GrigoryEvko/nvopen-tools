// Function: sub_36E1940
// Address: 0x36e1940
//
__int64 __fastcall sub_36E1940(__int64 a1, unsigned __int8 a2)
{
  __int64 v2; // rcx
  char v3; // dl
  __int64 v4; // r11
  int v5; // r8d
  unsigned int v6; // r10d
  char *v7; // r9
  char v8; // al
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v13; // r8
  __int64 v14; // r8
  int v15; // r9d
  int v16; // r12d
  _QWORD *v17; // [rsp-88h] [rbp-88h] BYREF
  __int16 v18; // [rsp-68h] [rbp-68h]
  _QWORD v19[4]; // [rsp-58h] [rbp-58h] BYREF
  char v20; // [rsp-38h] [rbp-38h]
  void *v21; // [rsp-30h] [rbp-30h] BYREF
  int v22; // [rsp-28h] [rbp-28h]
  void **v23; // [rsp-20h] [rbp-20h] BYREF

  v2 = *(unsigned int *)(a1 + 88);
  if ( !(_DWORD)v2 )
    BUG();
  v3 = *(_BYTE *)(a1 + 8) & 1;
  if ( v3 )
  {
    v4 = a1 + 16;
    v5 = 7;
  }
  else
  {
    v13 = *(unsigned int *)(a1 + 24);
    v4 = *(_QWORD *)(a1 + 16);
    if ( !(_DWORD)v13 )
      goto LABEL_13;
    v5 = v13 - 1;
  }
  v6 = v5 & (37 * a2);
  v7 = (char *)(v4 + 8LL * v6);
  v8 = *v7;
  if ( *v7 == a2 )
    goto LABEL_5;
  v15 = 1;
  while ( v8 != -1 )
  {
    v16 = v15 + 1;
    v6 = v5 & (v15 + v6);
    v7 = (char *)(v4 + 8LL * v6);
    v8 = *v7;
    if ( *v7 == a2 )
      goto LABEL_5;
    v15 = v16;
  }
  if ( v3 )
  {
    v14 = 64;
    goto LABEL_14;
  }
  v13 = *(unsigned int *)(a1 + 24);
LABEL_13:
  v14 = 8 * v13;
LABEL_14:
  v7 = (char *)(v4 + v14);
LABEL_5:
  v9 = 64;
  if ( !v3 )
    v9 = 8LL * *(unsigned int *)(a1 + 24);
  if ( v7 == (char *)(v4 + v9)
    || (v10 = *(_QWORD *)(a1 + 80), v11 = v10 + 8LL * *((unsigned int *)v7 + 1), v11 == v10 + 8 * v2) )
  {
    v20 = 1;
    v19[2] = &v23;
    v19[0] = "Could not find scope ID={}.";
    v22 = a2;
    v21 = &unk_4A3BF30;
    v19[1] = 27;
    v19[3] = 1;
    v23 = &v21;
    v18 = 263;
    v17 = v19;
    sub_C64D30((__int64)&v17, 1u);
  }
  return *(unsigned int *)(v11 + 4);
}
