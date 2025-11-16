// Function: sub_11A0AE0
// Address: 0x11a0ae0
//
__int64 __fastcall sub_11A0AE0(__int64 a1, unsigned int a2, __int64 *a3)
{
  __int64 v5; // rdx
  __int64 v6; // r12
  __int64 v7; // r15
  _BYTE *v8; // r14
  char v9; // r8
  __int64 result; // rax
  unsigned int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rbx
  __int64 v17; // rbx
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rdx
  _BYTE *v21; // rax
  __int64 v22; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v23; // [rsp+18h] [rbp-48h]
  __int64 v24; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v25; // [rsp+28h] [rbp-38h]

  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v5 = *(_QWORD *)(a1 - 8);
  else
    v5 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  v6 = 32LL * a2;
  v7 = *(_QWORD *)(v5 + v6);
  v8 = (_BYTE *)(v7 + 24);
  if ( *(_BYTE *)v7 != 17 )
  {
    v20 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v7 + 8) + 8LL) - 17;
    if ( (unsigned int)v20 > 1 )
      return 0;
    if ( *(_BYTE *)v7 > 0x15u )
      return 0;
    v21 = sub_AD7630(v7, 0, v20);
    if ( !v21 )
      return 0;
    v8 = v21 + 24;
    if ( *v21 != 17 )
      return 0;
  }
  if ( *((_DWORD *)v8 + 2) <= 0x40u )
  {
    result = 0;
    if ( (*(_QWORD *)v8 & ~*a3) == 0 )
      return result;
  }
  else
  {
    v9 = sub_C446F0((__int64 *)v8, a3);
    result = 0;
    if ( v9 )
      return result;
  }
  v11 = *((_DWORD *)v8 + 2);
  v23 = v11;
  if ( v11 > 0x40 )
  {
    sub_C43780((__int64)&v22, (const void **)v8);
    v11 = v23;
    if ( v23 > 0x40 )
    {
      sub_C43B90(&v22, a3);
      v11 = v23;
      v13 = v22;
      goto LABEL_9;
    }
    v12 = v22;
  }
  else
  {
    v12 = *(_QWORD *)v8;
  }
  v13 = *a3 & v12;
  v22 = v13;
LABEL_9:
  v25 = v11;
  v23 = 0;
  v14 = *(_QWORD *)(v7 + 8);
  v24 = v13;
  v15 = sub_AD8D80(v14, (__int64)&v24);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v16 = *(_QWORD *)(a1 - 8);
  else
    v16 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  v17 = v6 + v16;
  if ( *(_QWORD *)v17 )
  {
    v18 = *(_QWORD *)(v17 + 8);
    **(_QWORD **)(v17 + 16) = v18;
    if ( v18 )
      *(_QWORD *)(v18 + 16) = *(_QWORD *)(v17 + 16);
  }
  *(_QWORD *)v17 = v15;
  if ( v15 )
  {
    v19 = *(_QWORD *)(v15 + 16);
    *(_QWORD *)(v17 + 8) = v19;
    if ( v19 )
      *(_QWORD *)(v19 + 16) = v17 + 8;
    *(_QWORD *)(v17 + 16) = v15 + 16;
    *(_QWORD *)(v15 + 16) = v17;
  }
  if ( v25 > 0x40 && v24 )
    j_j___libc_free_0_0(v24);
  if ( v23 > 0x40 )
  {
    if ( v22 )
      j_j___libc_free_0_0(v22);
  }
  return 1;
}
