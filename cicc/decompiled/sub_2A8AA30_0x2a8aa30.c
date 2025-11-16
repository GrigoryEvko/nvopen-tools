// Function: sub_2A8AA30
// Address: 0x2a8aa30
//
unsigned __int64 __fastcall sub_2A8AA30(__int64 a1, char a2)
{
  __int64 v2; // rax
  unsigned int v3; // eax
  __int64 v4; // rcx
  _QWORD *v5; // rax
  unsigned __int64 result; // rax
  __int64 v7; // rbx
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 *v10; // rax
  __int64 v11; // r12
  __int64 *v12; // rax
  __int64 *v13; // rax
  int v14[9]; // [rsp+Ch] [rbp-24h] BYREF

  v2 = *(_QWORD *)(a1 - 32);
  if ( !v2 || *(_BYTE *)v2 || *(_QWORD *)(v2 + 24) != *(_QWORD *)(a1 + 80) )
    BUG();
  v3 = *(_DWORD *)(v2 + 36);
  if ( v3 == 8975 )
    goto LABEL_8;
  if ( v3 <= 0x230F )
  {
    if ( v3 - 8937 <= 1 )
    {
      v10 = (__int64 *)sub_BD5C60(a1);
      v14[0] = 1;
      v11 = sub_A77A40(v10, a2);
LABEL_19:
      v12 = (__int64 *)sub_BD5C60(a1);
      result = sub_A7B660((__int64 *)(a1 + 72), v12, v14, 1, v11);
      *(_QWORD *)(a1 + 72) = result;
      return result;
    }
    goto LABEL_26;
  }
  if ( v3 == 9553 )
  {
LABEL_22:
    v13 = (__int64 *)sub_BD5C60(a1);
    v14[0] = 2;
    v11 = sub_A77A40(v13, a2);
    goto LABEL_19;
  }
  if ( v3 != 9567 )
  {
    if ( v3 == 9549 )
      goto LABEL_22;
LABEL_26:
    BUG();
  }
LABEL_8:
  v4 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  v5 = *(_QWORD **)(v4 + 24);
  if ( *(_DWORD *)(v4 + 32) > 0x40u )
    v5 = (_QWORD *)*v5;
  result = sub_AD64C0(*(_QWORD *)(v4 + 8), (((a2 + 1) & 0x1F) << 13) | (unsigned int)v5 & 0xFFFC1FFF, 0);
  v7 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( *(_QWORD *)v7 )
  {
    v8 = *(_QWORD *)(v7 + 8);
    **(_QWORD **)(v7 + 16) = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = *(_QWORD *)(v7 + 16);
  }
  *(_QWORD *)v7 = result;
  if ( result )
  {
    v9 = *(_QWORD *)(result + 16);
    *(_QWORD *)(v7 + 8) = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = v7 + 8;
    *(_QWORD *)(v7 + 16) = result + 16;
    *(_QWORD *)(result + 16) = v7;
  }
  return result;
}
