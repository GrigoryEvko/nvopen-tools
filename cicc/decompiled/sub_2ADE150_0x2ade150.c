// Function: sub_2ADE150
// Address: 0x2ade150
//
void __fastcall sub_2ADE150(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  _QWORD *v8; // rdi
  _QWORD *v9; // rsi
  int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rax
  __int64 v16; // rcx
  int v17; // eax
  int v18; // edx
  unsigned int v19; // eax
  __int64 v20; // rsi
  int v21; // edi
  __int64 v22; // [rsp-60h] [rbp-60h] BYREF
  __int64 v23; // [rsp-58h] [rbp-58h] BYREF

  if ( *(_BYTE *)a3 == 63 && !(unsigned __int8)sub_D48480(*(_QWORD *)(**(_QWORD **)a1 + 416LL), a3, a3, a4) )
  {
    v7 = *(_QWORD *)(a1 + 8);
    v22 = a3;
    if ( *(_DWORD *)(v7 + 16) )
    {
      v16 = *(_QWORD *)(v7 + 8);
      v17 = *(_DWORD *)(v7 + 24);
      if ( v17 )
      {
        v18 = v17 - 1;
        v19 = (v17 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
        v20 = *(_QWORD *)(v16 + 8LL * v19);
        if ( a3 == v20 )
          return;
        v21 = 1;
        while ( v20 != -4096 )
        {
          v19 = v18 & (v21 + v19);
          v20 = *(_QWORD *)(v16 + 8LL * v19);
          if ( a3 == v20 )
            return;
          ++v21;
        }
      }
    }
    else
    {
      v8 = *(_QWORD **)(v7 + 32);
      v9 = &v8[*(unsigned int *)(v7 + 40)];
      if ( v9 != sub_2AA81A0(v8, (__int64)v9, &v22) )
        return;
    }
    v10 = sub_2AAA2B0(
            *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL),
            (__int64)a2,
            ***(_DWORD ***)(a1 + 16),
            *(_BYTE *)(**(_QWORD **)(a1 + 16) + 4LL));
    if ( *a2 == 62 && (v11 = *((_QWORD *)a2 - 8)) != 0 && a3 == v11 )
    {
      if ( v10 != 5 )
        goto LABEL_12;
    }
    else if ( v10 == 4 )
    {
      goto LABEL_12;
    }
    v15 = *(_QWORD *)(a3 + 16);
    if ( !v15 )
    {
LABEL_14:
      sub_2ADDD60(*(_QWORD *)(a1 + 24), &v22, v11, v12, v13, v14);
      return;
    }
    while ( 1 )
    {
      v11 = (unsigned int)**(unsigned __int8 **)(v15 + 24) - 61;
      if ( (unsigned __int8)(**(_BYTE **)(v15 + 24) - 61) > 1u )
        break;
      v15 = *(_QWORD *)(v15 + 8);
      if ( !v15 )
        goto LABEL_14;
    }
LABEL_12:
    sub_BED950((__int64)&v23, *(_QWORD *)(a1 + 32), a3);
  }
}
