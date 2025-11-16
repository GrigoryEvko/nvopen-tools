// Function: sub_D36C10
// Address: 0xd36c10
//
__int64 __fastcall sub_D36C10(_QWORD *a1)
{
  __int64 v1; // r12
  __int64 v2; // rax
  __int64 result; // rax
  __int64 v4; // rbx
  __int64 v5; // rdx
  __int64 v6; // r14
  __int64 v7; // rsi
  _QWORD *v8; // rax
  _QWORD *v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  char v13; // r8
  __int64 v14; // rax

  v1 = a1[3];
  if ( *(_QWORD *)(v1 + 8) == *(_QWORD *)(v1 + 16) )
  {
    v4 = *(_QWORD *)(**(_QWORD **)(v1 + 32) + 16LL);
    if ( !v4 )
      goto LABEL_18;
    while ( 1 )
    {
      v5 = *(_QWORD *)(v4 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v5 - 30) <= 0xAu )
        break;
      v4 = *(_QWORD *)(v4 + 8);
      if ( !v4 )
        goto LABEL_18;
    }
    v6 = 0;
    v7 = *(_QWORD *)(v5 + 40);
    if ( !*(_BYTE *)(v1 + 84) )
      goto LABEL_15;
LABEL_7:
    v8 = *(_QWORD **)(v1 + 64);
    v9 = &v8[*(unsigned int *)(v1 + 76)];
    if ( v8 != v9 )
    {
      while ( v7 != *v8 )
      {
        if ( v9 == ++v8 )
          goto LABEL_12;
      }
      ++v6;
    }
LABEL_12:
    while ( 1 )
    {
      v4 = *(_QWORD *)(v4 + 8);
      if ( !v4 )
        break;
      v10 = *(_QWORD *)(v4 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v10 - 30) <= 0xAu )
      {
        v7 = *(_QWORD *)(v10 + 40);
        if ( *(_BYTE *)(v1 + 84) )
          goto LABEL_7;
LABEL_15:
        if ( sub_C8CA60(v1 + 56, v7) )
          ++v6;
      }
    }
    if ( (_DWORD)v6 == 1 )
    {
      v12 = sub_DEF9D0(*a1, v7);
      v13 = sub_D96A50(v12);
      result = 1;
      if ( v13 )
      {
        v14 = sub_D364E0((__int64)a1, (__int64)"CantComputeNumberOfIterations", 29, 0);
        sub_B18290(v14, "could not determine number of loop iterations", 0x2Du);
        return 0;
      }
    }
    else
    {
LABEL_18:
      v11 = sub_D364E0((__int64)a1, (__int64)"CFGNotUnderstood", 16, 0);
      sub_B18290(v11, "loop control flow is not understood by analyzer", 0x2Fu);
      return 0;
    }
  }
  else
  {
    v2 = sub_D364E0((__int64)a1, (__int64)"NotInnerMostLoop", 16, 0);
    sub_B18290(v2, "loop is not the innermost loop", 0x1Eu);
    return 0;
  }
  return result;
}
