// Function: sub_228FB70
// Address: 0x228fb70
//
char __fastcall sub_228FB70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, char *a6, char *a7)
{
  __int64 v10; // rax
  _QWORD *v11; // r15
  __int64 v12; // rax
  _QWORD *v13; // rbx
  _QWORD *v14; // r14
  char v15; // al
  __int64 v16; // rdi
  __int64 *v17; // rax
  __int64 *v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 *v22; // r13
  __int64 *v23; // rax
  _QWORD *v24; // rax
  __int64 *v25; // r13
  __int64 *v26; // rax
  _QWORD *v27; // rax
  __int64 *v28; // rax
  __int64 *v29; // rax
  _QWORD *v30; // [rsp+8h] [rbp-48h]

  v10 = sub_D95540(a2);
  v11 = sub_228E360(a1, a6, v10);
  v12 = sub_D95540(a2);
  v13 = sub_228E360(a1, a7, v12);
  v30 = sub_DCC810(*(__int64 **)(a1 + 8), a5, a4, 0, 0);
  v14 = sub_DCC810(*(__int64 **)(a1 + 8), a4, a5, 0, 0);
  v15 = sub_DBED40(*(_QWORD *)(a1 + 8), a2);
  v16 = *(_QWORD *)(a1 + 8);
  if ( v15 )
  {
    if ( (unsigned __int8)sub_DBED40(v16, a3) )
    {
      if ( v11 )
      {
        v17 = sub_DCA690(*(__int64 **)(a1 + 8), a2, (__int64)v11, 0, 0);
        if ( sub_228DFC0(a1, 0x26u, (__int64)v30, (__int64)v17) )
          return 1;
      }
      if ( v13 )
      {
        v19 = sub_DCA690(*(__int64 **)(a1 + 8), a3, (__int64)v13, 0, 0);
        v20 = (__int64)v14;
        v21 = (__int64)v19;
        return sub_228DFC0(a1, 0x28u, v21, v20);
      }
      return 0;
    }
    if ( (unsigned __int8)sub_DBEC80(*(_QWORD *)(a1 + 8), a3) )
    {
      if ( v11 )
      {
        if ( v13 )
        {
          v25 = sub_DCA690(*(__int64 **)(a1 + 8), a2, (__int64)v11, 0, 0);
          v26 = sub_DCA690(*(__int64 **)(a1 + 8), a3, (__int64)v13, 0, 0);
          v27 = sub_DCC810(*(__int64 **)(a1 + 8), (__int64)v25, (__int64)v26, 0, 0);
          if ( sub_228DFC0(a1, 0x26u, (__int64)v30, (__int64)v27) )
            return 1;
        }
      }
      return sub_DBEC00(*(_QWORD *)(a1 + 8), (__int64)v30);
    }
    return 0;
  }
  if ( !(unsigned __int8)sub_DBEC80(v16, a2) )
    return 0;
  if ( !(unsigned __int8)sub_DBED40(*(_QWORD *)(a1 + 8), a3) )
  {
    if ( (unsigned __int8)sub_DBEC80(*(_QWORD *)(a1 + 8), a3) )
    {
      if ( v11 )
      {
        v28 = sub_DCA690(*(__int64 **)(a1 + 8), a2, (__int64)v11, 0, 0);
        if ( sub_228DFC0(a1, 0x26u, (__int64)v28, (__int64)v30) )
          return 1;
      }
      if ( v13 )
      {
        v29 = sub_DCA690(*(__int64 **)(a1 + 8), a3, (__int64)v13, 0, 0);
        v21 = (__int64)v14;
        v20 = (__int64)v29;
        return sub_228DFC0(a1, 0x28u, v21, v20);
      }
    }
    return 0;
  }
  if ( v11 )
  {
    if ( v13 )
    {
      v22 = sub_DCA690(*(__int64 **)(a1 + 8), a2, (__int64)v11, 0, 0);
      v23 = sub_DCA690(*(__int64 **)(a1 + 8), a3, (__int64)v13, 0, 0);
      v24 = sub_DCC810(*(__int64 **)(a1 + 8), (__int64)v22, (__int64)v23, 0, 0);
      if ( sub_228DFC0(a1, 0x26u, (__int64)v24, (__int64)v30) )
        return 1;
    }
  }
  return sub_DBEDC0(*(_QWORD *)(a1 + 8), (__int64)v30);
}
