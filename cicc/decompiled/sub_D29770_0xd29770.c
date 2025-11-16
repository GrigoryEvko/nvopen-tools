// Function: sub_D29770
// Address: 0xd29770
//
unsigned __int64 __fastcall sub_D29770(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  bool v9; // zf
  unsigned __int64 v10; // r13
  unsigned int v11; // esi
  __int64 v12; // r14
  __int64 v13; // rdi
  int v14; // r11d
  __int64 *v15; // r9
  unsigned int v16; // ecx
  __int64 *v17; // rax
  __int64 v18; // rdx
  unsigned __int64 *v19; // rax
  int v21; // edx
  int v22; // eax
  __int64 v23; // [rsp+0h] [rbp-30h] BYREF
  __int64 *v24; // [rsp+8h] [rbp-28h] BYREF

  v2 = a2;
  v4 = sub_D29010(a1, a2);
  v9 = *(_BYTE *)(v4 + 104) == 0;
  *(_QWORD *)(v4 + 16) = -1;
  v10 = v4;
  if ( !v9 )
  {
    v11 = *(_DWORD *)(a1 + 120);
    v23 = v2;
    v12 = a1 + 96;
    if ( v11 )
      goto LABEL_3;
LABEL_7:
    ++*(_QWORD *)(a1 + 96);
    v24 = 0;
    goto LABEL_8;
  }
  v12 = a1 + 96;
  sub_D29180(v4, a2, v5, v6, v7, v8);
  v11 = *(_DWORD *)(a1 + 120);
  v23 = v2;
  if ( !v11 )
    goto LABEL_7;
LABEL_3:
  v13 = *(_QWORD *)(a1 + 104);
  v14 = 1;
  v15 = 0;
  v16 = (v11 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
  v17 = (__int64 *)(v13 + 16LL * v16);
  v18 = *v17;
  if ( v2 == *v17 )
  {
LABEL_4:
    v19 = (unsigned __int64 *)(v17 + 1);
    goto LABEL_5;
  }
  while ( v18 != -4096 )
  {
    if ( !v15 && v18 == -8192 )
      v15 = v17;
    v16 = (v11 - 1) & (v14 + v16);
    v17 = (__int64 *)(v13 + 16LL * v16);
    v18 = *v17;
    if ( v2 == *v17 )
      goto LABEL_4;
    ++v14;
  }
  if ( !v15 )
    v15 = v17;
  v22 = *(_DWORD *)(a1 + 112);
  ++*(_QWORD *)(a1 + 96);
  v21 = v22 + 1;
  v24 = v15;
  if ( 4 * (v22 + 1) < 3 * v11 )
  {
    if ( v11 - *(_DWORD *)(a1 + 116) - v21 > v11 >> 3 )
      goto LABEL_20;
    goto LABEL_9;
  }
LABEL_8:
  v11 *= 2;
LABEL_9:
  sub_D25040(v12, v11);
  sub_D24A00(v12, &v23, &v24);
  v2 = v23;
  v15 = v24;
  v21 = *(_DWORD *)(a1 + 112) + 1;
LABEL_20:
  *(_DWORD *)(a1 + 112) = v21;
  if ( *v15 != -4096 )
    --*(_DWORD *)(a1 + 116);
  *v15 = v2;
  v19 = (unsigned __int64 *)(v15 + 1);
  v15[1] = 0;
LABEL_5:
  *v19 = v10;
  return v10;
}
