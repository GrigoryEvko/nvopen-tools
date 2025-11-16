// Function: sub_E215C0
// Address: 0xe215c0
//
__int64 __fastcall sub_E215C0(__int64 a1)
{
  _QWORD *v1; // rax
  unsigned __int64 v2; // rdx
  __int64 result; // rax
  __int64 *v4; // rax
  __int64 *v5; // r12
  __int64 v6; // rdx

  v1 = *(_QWORD **)(a1 + 16);
  v2 = (*v1 + v1[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
  v1[1] = v2 - *v1 + 32;
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) <= *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
  {
    result = 0;
    if ( !v2 )
      return result;
    result = v2;
    goto LABEL_4;
  }
  v4 = (__int64 *)sub_22077B0(32);
  v5 = v4;
  if ( v4 )
  {
    *v4 = 0;
    v4[1] = 0;
    v4[2] = 0;
    v4[3] = 0;
  }
  result = sub_2207820(4096);
  v6 = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(a1 + 16) = v5;
  *v5 = result;
  v5[3] = v6;
  v5[2] = 4096;
  v5[1] = 32;
  if ( result )
  {
LABEL_4:
    *(_DWORD *)(result + 8) = 9;
    *(_QWORD *)(result + 16) = 0;
    *(_QWORD *)(result + 24) = 0;
    *(_QWORD *)result = &unk_49E1000;
  }
  return result;
}
