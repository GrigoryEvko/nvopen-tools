// Function: sub_2AB8E90
// Address: 0x2ab8e90
//
__int64 __fastcall sub_2AB8E90(__int64 a1)
{
  unsigned __int64 v1; // r14
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r15
  signed __int64 v7; // rax
  __int64 v8; // r13
  __int64 *v9; // rbx
  __int64 v10; // rax
  __int64 *v11; // r13
  __int64 v12; // rsi
  __int64 v13; // r9
  __int64 *v14; // r8
  __int64 v15; // r9
  __int64 *v16; // r8
  __int64 *v17; // rax
  __int64 v18; // r9
  __int64 *v19; // r8
  unsigned __int64 v20; // rax
  __int64 v21; // r9
  __int64 *v22; // r8
  unsigned __int8 *v23[7]; // [rsp+8h] [rbp-38h] BYREF

  LODWORD(v1) = *(unsigned __int8 *)(a1 + 113);
  if ( (_BYTE)v1 )
  {
    LODWORD(v1) = *(unsigned __int8 *)(a1 + 112);
    return (unsigned int)v1;
  }
  *(_WORD *)(a1 + 112) = 256;
  if ( (unsigned __int8)sub_DFE610(*(_QWORD *)(a1 + 448)) || byte_500DD68 )
  {
    if ( !*(_DWORD *)(*(_QWORD *)(a1 + 496) + 88LL) )
    {
      v13 = *(_QWORD *)(a1 + 416);
      v14 = *(__int64 **)(a1 + 480);
      v23[0] = 0;
      sub_2AB8CE0(
        "Scalable vectorization is explicitly disabled",
        0x2Du,
        (__int64)"ScalableVectorizationDisabled",
        29,
        v14,
        v13,
        v23);
      sub_9C6650(v23);
      return (unsigned int)v1;
    }
    v3 = *(_QWORD *)(a1 + 440);
    BYTE4(v23[0]) = 1;
    LODWORD(v23[0]) = -1;
    v4 = *(_QWORD *)(v3 + 112);
    v5 = 184LL * *(unsigned int *)(v3 + 120);
    v6 = v4 + v5;
    v7 = 0xD37A6F4DE9BD37A7LL * (v5 >> 3);
    if ( v7 >> 2 )
    {
      v8 = v4 + 736 * (v7 >> 2);
      while ( (unsigned __int8)sub_DFE240(*(__int64 **)(a1 + 448), v4 + 8, (__int64)v23[0]) )
      {
        if ( !(unsigned __int8)sub_DFE240(*(__int64 **)(a1 + 448), v4 + 192, (__int64)v23[0]) )
        {
          v4 += 184;
          break;
        }
        if ( !(unsigned __int8)sub_DFE240(*(__int64 **)(a1 + 448), v4 + 376, (__int64)v23[0]) )
        {
          v4 += 368;
          break;
        }
        if ( !(unsigned __int8)sub_DFE240(*(__int64 **)(a1 + 448), v4 + 560, (__int64)v23[0]) )
        {
          v4 += 552;
          break;
        }
        v4 += 736;
        if ( v8 == v4 )
        {
          v7 = 0xD37A6F4DE9BD37A7LL * ((v6 - v4) >> 3);
          goto LABEL_37;
        }
      }
LABEL_14:
      if ( v6 != v4 )
      {
        v18 = *(_QWORD *)(a1 + 416);
        v19 = *(__int64 **)(a1 + 480);
        v23[0] = 0;
        sub_2AB8CE0(
          "Scalable vectorization not supported for the reduction operations found in this loop.",
          0x55u,
          (__int64)"ScalableVFUnfeasible",
          20,
          v19,
          v18,
          v23);
        sub_9C6650(v23);
        return (unsigned int)v1;
      }
      goto LABEL_15;
    }
LABEL_37:
    if ( v7 != 2 )
    {
      if ( v7 != 3 )
      {
        if ( v7 != 1 )
          goto LABEL_15;
        goto LABEL_40;
      }
      if ( !(unsigned __int8)sub_DFE240(*(__int64 **)(a1 + 448), v4 + 8, (__int64)v23[0]) )
        goto LABEL_14;
      v4 += 184;
    }
    if ( !(unsigned __int8)sub_DFE240(*(__int64 **)(a1 + 448), v4 + 8, (__int64)v23[0]) )
      goto LABEL_14;
    v4 += 184;
LABEL_40:
    if ( !(unsigned __int8)sub_DFE240(*(__int64 **)(a1 + 448), v4 + 8, (__int64)v23[0]) )
      goto LABEL_14;
LABEL_15:
    v9 = *(__int64 **)(a1 + 840);
    if ( *(_BYTE *)(a1 + 860) )
      v10 = *(unsigned int *)(a1 + 852);
    else
      v10 = *(unsigned int *)(a1 + 848);
    v11 = &v9[v10];
    if ( v11 != v9 )
    {
      while ( 1 )
      {
        v12 = *v9;
        if ( (unsigned __int64)*v9 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v11 == ++v9 )
          goto LABEL_20;
      }
      while ( v11 != v9 )
      {
        if ( *(_BYTE *)(v12 + 8) != 7 )
        {
          LODWORD(v1) = sub_DFE280(*(_QWORD *)(a1 + 448));
          if ( !(_BYTE)v1 )
          {
            if ( v11 == v9 )
              break;
            v15 = *(_QWORD *)(a1 + 416);
            v16 = *(__int64 **)(a1 + 480);
            v23[0] = 0;
            sub_2AB8CE0(
              "Scalable vectorization is not supported for all element types found in this loop.",
              0x51u,
              (__int64)"ScalableVFUnfeasible",
              20,
              v16,
              v15,
              v23);
            sub_9C6650(v23);
            return (unsigned int)v1;
          }
        }
        v17 = v9 + 1;
        if ( v11 == v9 + 1 )
          break;
        while ( 1 )
        {
          v12 = *v17;
          v9 = v17;
          if ( (unsigned __int64)*v17 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v11 == ++v17 )
            goto LABEL_20;
        }
      }
    }
LABEL_20:
    if ( *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 440) + 56LL) + 16LL) + 216LL) == 0xFFFFFFFFLL
      || (v20 = sub_2AA7E40(*(_QWORD *)(a1 + 488), *(_QWORD *)(a1 + 448)), v1 = HIDWORD(v20), BYTE4(v20)) )
    {
      LODWORD(v1) = 1;
      *(_WORD *)(a1 + 112) = 257;
    }
    else
    {
      v21 = *(_QWORD *)(a1 + 416);
      v22 = *(__int64 **)(a1 + 480);
      v23[0] = 0;
      sub_2AB8CE0(
        "The target does not provide maximum vscale value for safe distance analysis.",
        0x4Cu,
        (__int64)"ScalableVFUnfeasible",
        20,
        v22,
        v21,
        v23);
      sub_9C6650(v23);
    }
  }
  return (unsigned int)v1;
}
