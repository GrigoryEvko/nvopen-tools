// Function: sub_F0A410
// Address: 0xf0a410
//
__int64 __fastcall sub_F0A410(__int64 a1, __int64 a2, char a3)
{
  _BYTE *v3; // r8
  int v4; // r13d
  __int64 v7; // r15
  __int64 v8; // rdx
  __int64 v9; // rax
  _BYTE *v10; // r15
  __int64 v11; // r13
  __int64 v13; // rdi
  __int64 v14; // r12
  int v15; // eax
  int v16; // esi
  __int64 v17; // r15
  __int64 v18; // r12
  __int64 v19; // rdx
  _BYTE *v20; // rax
  __int64 v21; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v22; // [rsp+8h] [rbp-48h]
  __int64 v23; // [rsp+10h] [rbp-40h]
  unsigned int v24; // [rsp+18h] [rbp-38h]

  v3 = *(_BYTE **)(a2 + (a3 == 0 ? 0x20 : 0) - 64);
  v4 = (a3 == 0) + 1;
  if ( *v3 != 17 )
    return 0;
  v7 = ((*(_DWORD *)(a1 + 4) & 0x7FFFFFFu) >> 1) - 1;
  sub_F08550(a1, 0, a1, v7, (__int64)v3);
  v9 = 32;
  if ( v8 != v7 && (_DWORD)v8 != -2 )
    v9 = 32LL * (unsigned int)(2 * v8 + 3);
  if ( *(_QWORD *)(*(_QWORD *)(a1 - 8) + 32LL) != *(_QWORD *)(*(_QWORD *)(a1 - 8) + v9) )
    return 0;
  v10 = *(_BYTE **)(a2 - 96);
  v11 = *(_QWORD *)(a2 + 32LL * (unsigned int)(3 - v4) - 96);
  if ( *v10 != 82 || v11 != *((_QWORD *)v10 - 8) )
    return 0;
  v13 = *((_QWORD *)v10 - 4);
  v14 = v13 + 24;
  if ( *(_BYTE *)v13 == 17 )
    goto LABEL_11;
  v19 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v13 + 8) + 8LL) - 17;
  if ( (unsigned int)v19 > 1 )
    return 0;
  if ( *(_BYTE *)v13 > 0x15u )
    return 0;
  v20 = sub_AD7630(v13, 0, v19);
  if ( !v20 || *v20 != 17 )
    return 0;
  v14 = (__int64)(v20 + 24);
LABEL_11:
  v15 = sub_B53900((__int64)v10);
  v16 = v15;
  if ( a3 )
    v16 = sub_B52870(v15);
  v17 = 0;
  sub_AB1A50((__int64)&v21, v16, v14);
  v18 = ((*(_DWORD *)(a1 + 4) & 0x7FFFFFFu) >> 1) - 1;
  while ( v18 != v17 )
  {
    if ( !sub_AB1B10((__int64)&v21, *(_QWORD *)(*(_QWORD *)(a1 - 8) + 32LL * (unsigned int)(2 * ++v17)) + 24LL) )
    {
      v11 = 0;
      break;
    }
  }
  if ( v24 > 0x40 && v23 )
    j_j___libc_free_0_0(v23);
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  return v11;
}
