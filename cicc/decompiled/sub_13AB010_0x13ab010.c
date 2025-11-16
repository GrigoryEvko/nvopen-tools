// Function: sub_13AB010
// Address: 0x13ab010
//
__int64 __fastcall sub_13AB010(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // r14
  char v15; // al
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // r13
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // [rsp+8h] [rbp-48h]

  v10 = sub_1456040(a2);
  v11 = sub_13A7AF0(a1, a6, v10);
  v12 = sub_1456040(a2);
  v13 = sub_13A7AF0(a1, a7, v12);
  v30 = sub_14806B0(*(_QWORD *)(a1 + 8), a5, a4, 0, 0);
  v14 = sub_14806B0(*(_QWORD *)(a1 + 8), a4, a5, 0, 0);
  v15 = sub_1477BC0(*(_QWORD *)(a1 + 8), a2);
  v16 = *(_QWORD *)(a1 + 8);
  if ( v15 )
  {
    if ( (unsigned __int8)sub_1477BC0(v16, a3) )
    {
      if ( v11 )
      {
        v17 = sub_13A5B60(*(_QWORD *)(a1 + 8), a2, v11, 0, 0);
        if ( (unsigned __int8)sub_13A7760(a1, 38, v30, v17) )
          return 1;
      }
      if ( v13 )
      {
        v19 = sub_13A5B60(*(_QWORD *)(a1 + 8), a3, v13, 0, 0);
        v20 = v14;
        v21 = v19;
        return sub_13A7760(a1, 40, v21, v20);
      }
      return 0;
    }
    if ( (unsigned __int8)sub_1477A90(*(_QWORD *)(a1 + 8), a3) )
    {
      if ( v11 )
      {
        if ( v13 )
        {
          v25 = sub_13A5B60(*(_QWORD *)(a1 + 8), a2, v11, 0, 0);
          v26 = sub_13A5B60(*(_QWORD *)(a1 + 8), a3, v13, 0, 0);
          v27 = sub_14806B0(*(_QWORD *)(a1 + 8), v25, v26, 0, 0);
          if ( (unsigned __int8)sub_13A7760(a1, 38, v30, v27) )
            return 1;
        }
      }
      return sub_1477B50(*(_QWORD *)(a1 + 8), v30);
    }
    return 0;
  }
  if ( !(unsigned __int8)sub_1477A90(v16, a2) )
    return 0;
  if ( !(unsigned __int8)sub_1477BC0(*(_QWORD *)(a1 + 8), a3) )
  {
    if ( (unsigned __int8)sub_1477A90(*(_QWORD *)(a1 + 8), a3) )
    {
      if ( v11 )
      {
        v28 = sub_13A5B60(*(_QWORD *)(a1 + 8), a2, v11, 0, 0);
        if ( (unsigned __int8)sub_13A7760(a1, 38, v28, v30) )
          return 1;
      }
      if ( v13 )
      {
        v29 = sub_13A5B60(*(_QWORD *)(a1 + 8), a3, v13, 0, 0);
        v21 = v14;
        v20 = v29;
        return sub_13A7760(a1, 40, v21, v20);
      }
    }
    return 0;
  }
  if ( v11 )
  {
    if ( v13 )
    {
      v22 = sub_13A5B60(*(_QWORD *)(a1 + 8), a2, v11, 0, 0);
      v23 = sub_13A5B60(*(_QWORD *)(a1 + 8), a3, v13, 0, 0);
      v24 = sub_14806B0(*(_QWORD *)(a1 + 8), v22, v23, 0, 0);
      if ( (unsigned __int8)sub_13A7760(a1, 38, v24, v30) )
        return 1;
    }
  }
  return sub_1477C30(*(_QWORD *)(a1 + 8), v30);
}
