// Function: sub_157F7D0
// Address: 0x157f7d0
//
__int64 __fastcall sub_157F7D0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v8; // rdx
  _QWORD *v9; // rax

  v2 = sub_157EBA0(a2);
  if ( (*(_QWORD *)(v2 + 48) || *(__int16 *)(v2 + 18) < 0)
    && (v3 = sub_1625790(v2, 24), (v4 = v3) != 0)
    && (v5 = sub_161E970(*(_QWORD *)(v3 - 8LL * *(unsigned int *)(v3 + 8))), v6 == 18)
    && !(*(_QWORD *)v5 ^ 0x6165685F706F6F6CLL | *(_QWORD *)(v5 + 8) ^ 0x676965775F726564LL)
    && *(_WORD *)(v5 + 16) == 29800 )
  {
    v8 = *(_QWORD *)(*(_QWORD *)(v4 + 8 * (1LL - *(unsigned int *)(v4 + 8))) + 136LL);
    v9 = *(_QWORD **)(v8 + 24);
    if ( *(_DWORD *)(v8 + 32) > 0x40u )
      v9 = (_QWORD *)*v9;
    *(_BYTE *)(a1 + 8) = 1;
    *(_QWORD *)a1 = v9;
  }
  else
  {
    *(_BYTE *)(a1 + 8) = 0;
  }
  return a1;
}
