// Function: sub_161FFF0
// Address: 0x161fff0
//
__int64 *__fastcall sub_161FFF0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  char v4; // r8
  __int64 v5; // r15
  __int64 v6; // r14
  __int64 v7; // r13
  int v8; // ebx
  void *v9; // r11
  __int64 v10; // rax
  size_t v11; // rdx
  int v12; // r10d
  __int64 *v13; // rdi
  __int64 v14; // rax
  int v16; // [rsp+Ch] [rbp-74h]
  int v17; // [rsp+10h] [rbp-70h]
  char v18; // [rsp+18h] [rbp-68h]
  __int64 v19; // [rsp+20h] [rbp-60h]
  int v20; // [rsp+28h] [rbp-58h]
  int v21; // [rsp+2Ch] [rbp-54h]
  __int64 v22; // [rsp+30h] [rbp-50h]
  __int64 v23; // [rsp+38h] [rbp-48h]
  int v24; // [rsp+48h] [rbp-38h] BYREF
  char v25; // [rsp+4Ch] [rbp-34h]

  v3 = *(unsigned int *)(a2 + 8);
  v4 = *(_BYTE *)(a2 + 56);
  v20 = *(_DWORD *)(a2 + 28);
  v5 = *(_QWORD *)(a2 + 8 * (4 - v3));
  if ( v4 )
    v16 = *(_DWORD *)(a2 + 52);
  v6 = a2;
  v19 = *(_QWORD *)(a2 + 40);
  v21 = *(_DWORD *)(a2 + 48);
  v22 = *(_QWORD *)(a2 + 32);
  v7 = *(_QWORD *)(a2 + 8 * (1 - v3));
  v8 = *(_DWORD *)(a2 + 24);
  v23 = *(_QWORD *)(a2 + 8 * (3 - v3));
  if ( *(_BYTE *)a2 != 15 )
    v6 = *(_QWORD *)(a2 - 8 * v3);
  v9 = *(void **)(a2 + 8 * (2 - v3));
  if ( v9 )
  {
    v18 = *(_BYTE *)(a2 + 56);
    v10 = sub_161E970(*(_QWORD *)(a2 + 8 * (2 - v3)));
    v4 = v18;
    v9 = (void *)v10;
  }
  else
  {
    v11 = 0;
  }
  v12 = *(unsigned __int16 *)(a2 + 2);
  v13 = (__int64 *)(*(_QWORD *)(a2 + 16) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a2 + 16) & 4) != 0 )
    v13 = (__int64 *)*v13;
  if ( v4 )
  {
    v25 = 1;
    v24 = v16;
  }
  else
  {
    v25 = 0;
  }
  v14 = 0;
  if ( v11 )
  {
    v17 = v12;
    v14 = sub_161FF10(v13, v9, v11);
    v12 = v17;
  }
  *a1 = sub_15BD310(v13, v12, v14, v6, v8, v7, v23, v22, v21, v19, &v24, v20, v5, 2u, 1);
  return a1;
}
