// Function: sub_3735F40
// Address: 0x3735f40
//
__int64 __fastcall sub_3735F40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rsi
  int v10; // edi
  __int64 v11; // rbx
  unsigned int v12; // eax
  __int64 *v13; // rdx
  __int64 *v14; // r14
  unsigned int v15; // ebx
  __int64 v17; // rbx
  char v18; // al
  __int64 v19; // rcx
  int v20; // eax
  _BYTE *v21; // [rsp+10h] [rbp-60h] BYREF
  __int64 v22; // [rsp+18h] [rbp-58h]
  _BYTE v23[80]; // [rsp+20h] [rbp-50h] BYREF

  *(_BYTE *)(a1 + 392) = 1;
  if ( (unsigned __int16)sub_3220AA0(*(_QWORD *)(a1 + 208)) > 4u )
  {
    v6 = *(unsigned int *)(a3 + 8);
    v21 = v23;
    v11 = *(_QWORD *)(a1 + 216);
    v22 = 0x200000000LL;
    if ( !(_DWORD)v6 )
      goto LABEL_6;
  }
  else
  {
    v9 = *(_QWORD *)(a1 + 408);
    if ( v9 )
    {
      v10 = *(_DWORD *)(a3 + 8);
      v11 = *(_QWORD *)(v9 + 216);
      v21 = v23;
      v22 = 0x200000000LL;
      if ( !v10 )
        goto LABEL_8;
    }
    else
    {
      v11 = *(_QWORD *)(a1 + 216);
      v22 = 0x200000000LL;
      v20 = *(_DWORD *)(a3 + 8);
      v21 = v23;
      if ( !v20 )
      {
        v9 = a1;
        goto LABEL_8;
      }
    }
  }
  sub_37352C0((__int64)&v21, (char **)a3, (__int64)&v21, v6, v7, v8);
LABEL_6:
  v9 = *(_QWORD *)(a1 + 408);
  if ( !v9 )
    v9 = a1;
LABEL_8:
  v12 = sub_3245420(v11, v9, (char *)&v21, v6, v7);
  v14 = v13;
  v15 = v12;
  if ( v21 != v23 )
    _libc_free((unsigned __int64)v21);
  if ( (unsigned __int16)sub_3220AA0(*(_QWORD *)(a1 + 208)) <= 4u )
  {
    v17 = *(_QWORD *)(*(_QWORD *)(sub_31DA6B0(*(_QWORD *)(a1 + 184)) + 160) + 16LL);
    v18 = sub_3734FE0(a1);
    v19 = *v14;
    if ( v18 )
      return sub_324AB90((__int64 *)a1, a2, 85, v19, v17);
    else
      return sub_324AC60((__int64 *)a1, a2, 85, v19, v17);
  }
  else
  {
    LODWORD(v21) = 65571;
    return sub_3249A20((__int64 *)a1, (unsigned __int64 **)(a2 + 8), 85, 65571, v15);
  }
}
