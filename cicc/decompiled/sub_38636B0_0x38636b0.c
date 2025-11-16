// Function: sub_38636B0
// Address: 0x38636b0
//
__int64 __fastcall sub_38636B0(__int64 *a1, __m128i a2, __m128i a3)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 result; // rax
  __int64 v6; // r14
  _QWORD *v7; // rax
  __int64 v8; // rbx
  int v9; // r13d
  __int64 v10; // rbx
  __int64 v11; // rbx
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // rax

  v3 = a1[3];
  if ( *(_QWORD *)(v3 + 16) == *(_QWORD *)(v3 + 8) )
  {
    v6 = *(_QWORD *)(**(_QWORD **)(v3 + 32) + 8LL);
    if ( !v6 )
      goto LABEL_16;
    while ( 1 )
    {
      v7 = sub_1648700(v6);
      if ( (unsigned __int8)(*((_BYTE *)v7 + 16) - 25) <= 9u )
        break;
      v6 = *(_QWORD *)(v6 + 8);
      if ( !v6 )
        goto LABEL_16;
    }
    v8 = v3 + 56;
    v9 = 0;
LABEL_8:
    v9 -= !sub_1377F70(v8, v7[5]) - 1;
    while ( 1 )
    {
      v6 = *(_QWORD *)(v6 + 8);
      if ( !v6 )
        break;
      v7 = sub_1648700(v6);
      if ( (unsigned __int8)(*((_BYTE *)v7 + 16) - 25) <= 9u )
        goto LABEL_8;
    }
    if ( v9 == 1 && sub_13F9E70(a1[3]) && (v10 = sub_13F9E70(a1[3]), v10 == sub_13FCB50(a1[3])) )
    {
      v11 = sub_1495DC0(*a1, a2, a3);
      v12 = sub_1456E90(*(_QWORD *)(*a1 + 112));
      result = 1;
      if ( v11 == v12 )
      {
        v13 = sub_3860590((__int64)a1, (__int64)"CantComputeNumberOfIterations", 29, 0);
        sub_15CAB20(v13, "could not determine number of loop iterations", 0x2Du);
        return 0;
      }
    }
    else
    {
LABEL_16:
      v14 = sub_3860590((__int64)a1, (__int64)"CFGNotUnderstood", 16, 0);
      sub_15CAB20(v14, "loop control flow is not understood by analyzer", 0x2Fu);
      return 0;
    }
  }
  else
  {
    v4 = sub_3860590((__int64)a1, (__int64)"NotInnerMostLoop", 16, 0);
    sub_15CAB20(v4, "loop is not the innermost loop", 0x1Eu);
    return 0;
  }
  return result;
}
