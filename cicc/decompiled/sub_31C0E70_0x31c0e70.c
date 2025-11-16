// Function: sub_31C0E70
// Address: 0x31c0e70
//
char __fastcall sub_31C0E70(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v5; // r8
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // rbx
  __int64 v9; // rcx
  __int64 v10; // r15
  unsigned __int8 v11; // r11
  unsigned __int8 v12; // si
  unsigned __int8 v13; // al
  __int64 v14; // rdx
  _QWORD *v15; // rcx
  bool v17; // al
  __int64 v21; // [rsp+18h] [rbp-58h]
  __int64 v22; // [rsp+30h] [rbp-40h]
  unsigned __int8 v23; // [rsp+38h] [rbp-38h]

  if ( a2 < (a3 - 1) / 2 )
  {
    v4 = a2;
    v5 = (a3 - 1) / 2;
    while ( 1 )
    {
      v7 = 2 * (v4 + 1);
      v8 = a1 + 16 * (v4 + 1);
      v6 = *(_QWORD *)v8;
      v9 = a1 + 8 * (v7 - 1);
      v10 = *(_QWORD *)(*(_QWORD *)v9 + 8LL);
      v11 = (unsigned int)**(unsigned __int8 **)(*(_QWORD *)(*(_QWORD *)v8 + 8LL) + 16LL) - 30 <= 0xA;
      v12 = (unsigned int)**(unsigned __int8 **)(v10 + 16) - 30 <= 0xA;
      if ( v11 != v12 )
        break;
      v21 = v5;
      v22 = *(_QWORD *)(*(_QWORD *)v8 + 8LL);
      v23 = sub_318B700(v22);
      v13 = sub_318B700(v10);
      v14 = v7 - 1;
      v15 = (_QWORD *)(a1 + 8 * (v7 - 1));
      v5 = v21;
      if ( v23 == v13 )
      {
        v17 = sub_B445A0(*(_QWORD *)(v10 + 16), *(_QWORD *)(v22 + 16));
        v14 = v7 - 1;
        v15 = (_QWORD *)(a1 + 8 * (v7 - 1));
        v5 = v21;
        if ( !v17 )
        {
LABEL_16:
          v6 = *(_QWORD *)v8;
          goto LABEL_5;
        }
      }
      else if ( v23 >= v13 )
      {
        goto LABEL_16;
      }
      v7 = v14;
      *(_QWORD *)(a1 + 8 * v4) = *v15;
      if ( v14 >= v5 )
        goto LABEL_11;
LABEL_6:
      v4 = v7;
    }
    if ( v11 > v12 )
    {
      v6 = *(_QWORD *)v9;
      --v7;
    }
LABEL_5:
    *(_QWORD *)(a1 + 8 * v4) = v6;
    if ( v7 >= v5 )
      goto LABEL_11;
    goto LABEL_6;
  }
  v7 = a2;
LABEL_11:
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == v7 )
  {
    *(_QWORD *)(a1 + 8 * v7) = *(_QWORD *)(a1 + 8 * (2 * v7 + 1));
    v7 = 2 * v7 + 1;
  }
  return sub_31BFEC0(a1, v7, a2, a4);
}
