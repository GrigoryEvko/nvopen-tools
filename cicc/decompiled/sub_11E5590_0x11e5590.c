// Function: sub_11E5590
// Address: 0x11e5590
//
__int64 __fastcall sub_11E5590(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  unsigned __int8 v4; // si
  __int64 v5; // rcx
  unsigned __int8 *v6; // rdi
  __int64 v7; // rdx
  __int64 *v8; // r15
  __int64 v9; // r12
  _QWORD *v10; // r13
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // rsi
  __int64 v15; // rdx
  _BYTE *v16; // rax
  _BYTE *v17; // rax
  __int64 v18; // [rsp+8h] [rbp-98h]
  __int64 v19[4]; // [rsp+10h] [rbp-90h] BYREF
  __int64 v20[4]; // [rsp+30h] [rbp-70h] BYREF
  void *v21[10]; // [rsp+50h] [rbp-50h] BYREF

  if ( !sub_B49E00(a2) )
    return 0;
  result = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v4 = *(_BYTE *)result;
  if ( *(_BYTE *)result == 13 )
    return result;
  v5 = 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v6 = *(unsigned __int8 **)(a2 + v5);
  v7 = *v6;
  if ( (_BYTE)v7 == 13 )
    return *(_QWORD *)(a2 + v5);
  v8 = (__int64 *)(result + 24);
  if ( v4 != 18 )
  {
    v15 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(result + 8) + 8LL) - 17;
    if ( (unsigned int)v15 > 1 )
      return 0;
    if ( v4 > 0x15u )
      return 0;
    v17 = sub_AD7630(*(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), 0, v15);
    if ( !v17 || *v17 != 18 )
      return 0;
    v8 = (__int64 *)(v17 + 24);
    v6 = *(unsigned __int8 **)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
    v7 = *v6;
  }
  v9 = (__int64)(v6 + 24);
  if ( (_BYTE)v7 != 18 )
  {
    if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v6 + 1) + 8LL) - 17 <= 1 && (unsigned __int8)v7 <= 0x15u )
    {
      v16 = sub_AD7630((__int64)v6, 0, v7);
      if ( v16 )
      {
        if ( *v16 == 18 )
        {
          v9 = (__int64)(v16 + 24);
          goto LABEL_8;
        }
      }
    }
    return 0;
  }
LABEL_8:
  v10 = sub_C33340();
  if ( (_QWORD *)*v8 == v10 )
    sub_C3C790(v19, (_QWORD **)v8);
  else
    sub_C33EB0(v19, v8);
  if ( (_QWORD *)v19[0] == v10 )
    sub_C3D820(v19, v9, 1u);
  else
    sub_C3B1F0((__int64)v19, v9, 1);
  v14 = sub_BCAC60(*(_QWORD *)(a2 + 8), v9, v11, v12, v13);
  if ( (_QWORD *)v14 == v10 )
    sub_C3C500(v21, (__int64)v10);
  else
    sub_C373C0(v21, v14);
  if ( v10 == v21[0] )
    sub_C3CEB0(v21, 0);
  else
    sub_C37310((__int64)v21, 0);
  sub_969CF0(v20, v19, (__int64 *)v21);
  sub_91D830(v21);
  v18 = sub_AD8F10(*(_QWORD *)(a2 + 8), v20);
  sub_91D830(v20);
  sub_91D830(v19);
  return v18;
}
