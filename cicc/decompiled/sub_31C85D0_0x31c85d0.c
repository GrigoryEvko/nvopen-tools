// Function: sub_31C85D0
// Address: 0x31c85d0
//
__int64 __fastcall sub_31C85D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  _BYTE *v3; // r15
  __int64 v5; // rax
  unsigned __int8 *v6; // rbx
  __int64 v7; // rax
  unsigned __int8 *v8; // rax
  unsigned int v9; // r13d
  unsigned int v10; // ebx
  unsigned __int64 v11; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-48h]
  unsigned __int64 v13; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v14; // [rsp+28h] [rbp-38h]

  v2 = 0;
  v3 = *(_BYTE **)(a2 - 64);
  if ( *v3 != 61 )
    return v2;
  v5 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
  if ( (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17 <= 1 )
    v5 = **(_QWORD **)(v5 + 16);
  v12 = sub_AE2980(*(_QWORD *)(a1 + 24), *(_DWORD *)(v5 + 8) >> 8)[3];
  if ( v12 > 0x40 )
    sub_C43690((__int64)&v11, 0, 0);
  else
    v11 = 0;
  v6 = sub_BD45C0(*(unsigned __int8 **)(a2 - 32), *(_QWORD *)(a1 + 24), (__int64)&v11, 0, 0, 0, 0, 0);
  if ( *v6 != 60 )
    v6 = 0;
  v7 = *(_QWORD *)(*((_QWORD *)v3 - 4) + 8LL);
  if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 <= 1 )
    v7 = **(_QWORD **)(v7 + 16);
  v14 = sub_AE2980(*(_QWORD *)(a1 + 24), *(_DWORD *)(v7 + 8) >> 8)[3];
  if ( v14 > 0x40 )
    sub_C43690((__int64)&v13, 0, 0);
  else
    v13 = 0;
  v8 = sub_BD45C0(*((unsigned __int8 **)v3 - 4), *(_QWORD *)(a1 + 24), (__int64)&v13, 0, 0, 0, 0, 0);
  v9 = v12;
  v2 = (__int64)v8;
  if ( *v8 != 22 )
    v2 = 0;
  if ( v12 <= 0x40 )
  {
    if ( v11 )
      goto LABEL_30;
  }
  else if ( v9 != (unsigned int)sub_C444A0((__int64)&v11) )
  {
    goto LABEL_30;
  }
  if ( !v6 || !v2 )
    goto LABEL_30;
  v10 = v14;
  if ( v14 <= 0x40 )
  {
    if ( v13 )
    {
LABEL_24:
      if ( v9 > 0x40 && v11 )
        j_j___libc_free_0_0(v11);
      return 0;
    }
  }
  else if ( (unsigned int)sub_C444A0((__int64)&v13) != v10 )
  {
    goto LABEL_21;
  }
  if ( !(unsigned __int8)sub_B2D680(v2) )
  {
LABEL_30:
    if ( v14 <= 0x40 )
    {
LABEL_23:
      v9 = v12;
      goto LABEL_24;
    }
LABEL_21:
    if ( v13 )
      j_j___libc_free_0_0(v13);
    goto LABEL_23;
  }
  if ( v14 > 0x40 && v13 )
    j_j___libc_free_0_0(v13);
  if ( v12 > 0x40 && v11 )
    j_j___libc_free_0_0(v11);
  return v2;
}
