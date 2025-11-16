// Function: sub_2E0B2D0
// Address: 0x2e0b2d0
//
__int64 __fastcall sub_2E0B2D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  _BYTE *v3; // rax
  _BYTE *v4; // rax
  _BYTE *v5; // rax
  __int64 result; // rax
  _BYTE *v7; // rdx
  _QWORD v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = a1;
  v3 = *(_BYTE **)(a1 + 32);
  if ( (unsigned __int64)v3 >= *(_QWORD *)(a1 + 24) )
  {
    v2 = sub_CB5D20(a1, 91);
  }
  else
  {
    *(_QWORD *)(a1 + 32) = v3 + 1;
    *v3 = 91;
  }
  v8[0] = *(_QWORD *)a2;
  sub_2FAD600(v8, v2);
  v4 = *(_BYTE **)(v2 + 32);
  if ( (unsigned __int64)v4 >= *(_QWORD *)(v2 + 24) )
  {
    v2 = sub_CB5D20(v2, 44);
  }
  else
  {
    *(_QWORD *)(v2 + 32) = v4 + 1;
    *v4 = 44;
  }
  v8[0] = *(_QWORD *)(a2 + 8);
  sub_2FAD600(v8, v2);
  v5 = *(_BYTE **)(v2 + 32);
  if ( (unsigned __int64)v5 >= *(_QWORD *)(v2 + 24) )
  {
    v2 = sub_CB5D20(v2, 58);
  }
  else
  {
    *(_QWORD *)(v2 + 32) = v5 + 1;
    *v5 = 58;
  }
  result = sub_CB59D0(v2, **(unsigned int **)(a2 + 16));
  v7 = *(_BYTE **)(result + 32);
  if ( (unsigned __int64)v7 >= *(_QWORD *)(result + 24) )
    return sub_CB5D20(result, 41);
  *(_QWORD *)(result + 32) = v7 + 1;
  *v7 = 41;
  return result;
}
