// Function: sub_1649990
// Address: 0x1649990
//
__int64 __fastcall sub_1649990(__int64 a1)
{
  __int64 v1; // rdx
  unsigned __int64 v2; // rax
  _QWORD *v3; // r12
  __int64 result; // rax
  __int64 v5; // rcx
  unsigned __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 *v8; // rax

  v1 = *(_QWORD *)(a1 + 8);
  v2 = *(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFFCLL;
  *(_QWORD *)v2 = v1;
  if ( v1 )
    *(_QWORD *)(v1 + 16) = *(_QWORD *)(v1 + 16) & 3LL | v2;
  v3 = sub_1648700(a1);
  result = *(v3 - 3);
  if ( *(_DWORD *)(result + 36) == 4 )
  {
    if ( (unsigned int)sub_1648720(a1) )
    {
      result = sub_1599EF0(**(__int64 ****)a1);
      if ( *(_QWORD *)a1 )
        goto LABEL_7;
    }
    else
    {
      v8 = (__int64 *)sub_16498A0((__int64)v3);
      result = sub_159C4F0(v8);
      if ( *(_QWORD *)a1 )
      {
LABEL_7:
        v5 = *(_QWORD *)(a1 + 8);
        v6 = *(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v6 = v5;
        if ( v5 )
          *(_QWORD *)(v5 + 16) = *(_QWORD *)(v5 + 16) & 3LL | v6;
      }
    }
    *(_QWORD *)a1 = result;
    if ( result )
    {
      v7 = *(_QWORD *)(result + 8);
      *(_QWORD *)(a1 + 8) = v7;
      if ( v7 )
        *(_QWORD *)(v7 + 16) = (a1 + 8) | *(_QWORD *)(v7 + 16) & 3LL;
      *(_QWORD *)(a1 + 16) = (result + 8) | *(_QWORD *)(a1 + 16) & 3LL;
      *(_QWORD *)(result + 8) = a1;
    }
  }
  return result;
}
