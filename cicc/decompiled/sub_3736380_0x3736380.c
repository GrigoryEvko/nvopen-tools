// Function: sub_3736380
// Address: 0x3736380
//
__int64 __fastcall sub_3736380(__int64 *a1, __int64 a2, unsigned __int64 a3)
{
  __int64 v5; // r12
  unsigned __int8 v6; // al
  __int64 v7; // rdx
  __int64 v8; // rdi
  const void *v9; // rax
  size_t v10; // rdx
  int v11; // r8d
  unsigned __int8 v12; // al
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 result; // rax

  v5 = *(_QWORD *)(a2 + 8);
  v6 = *(_BYTE *)(v5 - 16);
  if ( (v6 & 2) != 0 )
    v7 = *(_QWORD *)(v5 - 32);
  else
    v7 = v5 - 16 - 8LL * ((v6 >> 2) & 0xF);
  v8 = *(_QWORD *)(v7 + 8);
  if ( !v8 )
    goto LABEL_7;
  v9 = (const void *)sub_B91420(v8);
  if ( v10 )
    sub_324AD70(a1, a3, 3, v9, v10);
  v5 = *(_QWORD *)(a2 + 8);
  if ( v5 )
  {
LABEL_7:
    v11 = *(_DWORD *)(v5 + 4) >> 3;
    if ( v11 )
    {
      sub_3249A20(a1, (unsigned __int64 **)(a3 + 8), 136, 65551, v11 & 0x1FFFFFFF);
      v12 = *(_BYTE *)(v5 - 16);
      if ( (v12 & 2) == 0 )
        goto LABEL_9;
    }
    else
    {
      v12 = *(_BYTE *)(v5 - 16);
      if ( (v12 & 2) == 0 )
      {
LABEL_9:
        v13 = v5 - 16 - 8LL * ((v12 >> 2) & 0xF);
LABEL_10:
        sub_324CC60(a1, a3, *(_QWORD *)(v13 + 32));
        goto LABEL_11;
      }
    }
    v13 = *(_QWORD *)(v5 - 32);
    goto LABEL_10;
  }
LABEL_11:
  sub_3249D10(a1, a3, v5);
  v14 = sub_321DF00(a2);
  sub_32495E0(a1, a3, v14, 73);
  if ( (*(_BYTE *)(*(_QWORD *)(a2 + 8) + 24LL) & 0x40) != 0 )
    return sub_3249FA0(a1, a3, 52);
  result = sub_321DF00(a2);
  if ( (*(_BYTE *)(result + 20) & 0x40) != 0 )
    return sub_3249FA0(a1, a3, 52);
  return result;
}
