// Function: sub_24E9010
// Address: 0x24e9010
//
__int64 __fastcall sub_24E9010(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // rdx
  int v3; // eax
  __int64 result; // rax
  int v5; // eax
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 *v9; // rbx
  __int64 *i; // r12
  _QWORD *v11; // [rsp+8h] [rbp-48h]
  __int64 v12[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = *(_QWORD *)(a1 + 24);
  v3 = *(_DWORD *)(v2 + 280);
  if ( v3 )
  {
    result = (unsigned int)(v3 - 1);
    if ( (unsigned int)result <= 2 )
      return result;
    goto LABEL_9;
  }
  v5 = *(_DWORD *)(a1 + 32);
  if ( v5 > 2 )
  {
    if ( (unsigned int)(v5 - 3) <= 1 )
      goto LABEL_7;
LABEL_17:
    BUG();
  }
  if ( v5 > 0 )
  {
    v6 = 1;
    goto LABEL_8;
  }
  if ( v5 )
    goto LABEL_17;
LABEL_7:
  v6 = 0;
LABEL_8:
  v7 = sub_BCB2B0(*(_QWORD **)(a1 + 112));
  v8 = sub_ACD640(v7, v6, 0);
  v2 = *(_QWORD *)(a1 + 24);
  v1 = v8;
LABEL_9:
  v9 = *(__int64 **)(v2 + 120);
  result = *(unsigned int *)(v2 + 128);
  for ( i = &v9[result]; i != v9; ++v9 )
  {
    result = *v9;
    if ( *(_QWORD *)(a1 + 296) != *v9 )
    {
      v12[0] = *v9;
      v11 = (_QWORD *)sub_24E84F0(a1 + 200, v12)[2];
      sub_BD84D0((__int64)v11, v1);
      result = sub_B43D60(v11);
    }
  }
  return result;
}
